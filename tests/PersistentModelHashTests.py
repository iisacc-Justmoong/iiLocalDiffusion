"""Persist only verified hashes; changes and unusable caches must fail safe."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'reference/diffusers'))
import weight_files as w
class PersistentModelHashTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'build',prefix='persistent-hash-');self.addCleanup(self.tmp.cleanup)
  self.root=Path(self.tmp.name);self.model=self.root/'model.safetensors';self.model.write_bytes(b'original')
  self.env=patch.dict(os.environ,{'IILD_MODEL_HASH_CACHE':str(self.root/'hashes.sqlite3')});self.env.start();self.addCleanup(self.env.stop)
  w.clear_model_hash_cache()
 def test_new_process_reuses_verified_digest_without_reading_model(self):
  expected=w.cached_model_sha256(self.model)
  script='import sys; sys.path.insert(0,sys.argv[1]); import weight_files as w; from pathlib import Path; from unittest.mock import patch\nwith patch.object(w,"file_sha256",side_effect=AssertionError("full reread")): print(w.cached_model_sha256(Path(sys.argv[2])))'
  result=subprocess.run([sys.executable,'-c',script,str(ROOT/'reference/diffusers'),str(self.model)],capture_output=True,text=True,check=True)
  self.assertEqual(result.stdout.strip(),expected)
 def test_changed_file_with_restored_mtime_is_rehashed(self):
  old=self.model.stat();digest=w.cached_model_sha256(self.model);w.clear_model_hash_cache()
  self.model.write_bytes(b'modified');os.utime(self.model,ns=(old.st_atime_ns,old.st_mtime_ns))
  self.assertNotEqual(w.cached_model_sha256(self.model),digest)
  self.assertEqual(w.model_hash_statistics()['model_hashes'],1)
 def test_replacement_is_rehashed(self):
  w.cached_model_sha256(self.model);w.clear_model_hash_cache()
  replacement=self.root/'new';replacement.write_bytes(b'modified');replacement.replace(self.model)
  self.assertEqual(w.cached_model_sha256(self.model),hashlib.sha256(b'modified').hexdigest())
 def test_mutation_during_hash_does_not_persist(self):
  original=w.file_sha256
  def changing(path):
   result=original(path);path.write_bytes(b'modified');return result
  with patch.object(w,'file_sha256',side_effect=changing),self.assertRaises(RuntimeError):w.cached_model_sha256(self.model)
  w.clear_model_hash_cache()
  self.assertEqual(w.cached_model_sha256(self.model),hashlib.sha256(b'modified').hexdigest())
  self.assertEqual(w.model_hash_statistics()['model_hashes'],1)
 def test_corrupt_cache_falls_back_to_full_hash(self):
  (self.root/'hashes.sqlite3').write_bytes(b'not sqlite')
  self.assertEqual(w.cached_model_sha256(self.model),hashlib.sha256(b'original').hexdigest())
 def test_unwritable_cache_falls_back_to_full_hash(self):
  with patch.dict(os.environ,{'IILD_MODEL_HASH_CACHE':str(self.model/'not-a-directory')}):
   self.assertEqual(w.cached_model_sha256(self.model),hashlib.sha256(b'original').hexdigest())
if __name__=='__main__':unittest.main()
