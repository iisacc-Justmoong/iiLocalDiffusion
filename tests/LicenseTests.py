#!/usr/bin/env python3
"""Assert the first-party iiLocalDiffusion grant is Apache-2.0."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class LicenseTests(unittest.TestCase):
    def test_root_license_is_apache_2(self):
        text = (ROOT / "LICENSE").read_text(encoding="utf-8")
        self.assertIn("                                 Apache License\n", text[:200])
        self.assertIn("Version 2.0, January 2004", text)
        self.assertIn("http://www.apache.org/licenses/LICENSE-2.0", text)
        self.assertNotIn("Permission is hereby granted", text)
        self.assertNotIn("MIT License", text)

    def test_notice_names_the_copyright_holder(self):
        text = " ".join((ROOT / "NOTICE").read_text(encoding="utf-8").split())
        self.assertIn("Copyright (c) 2026 iisacc", text)
        self.assertIn("Apache License, Version 2.0", text)
        self.assertIn("GNU Affero General Public License (AGPL)", text)

    def test_readme_and_korean_statement_use_apache_spdx(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("SPDX-License-Identifier: Apache-2.0", readme)
        self.assertNotIn("SPDX-License-Identifier: MIT", readme)
        self.assertIn("[Apache License, Version 2.0](LICENSE)", readme)
        korean = (ROOT / "docs" / "License.md").read_text(encoding="utf-8")
        self.assertIn("SPDX-License-Identifier: Apache-2.0", korean)
        self.assertIn("Apache License 2.0", korean)
        self.assertIn("Copyright (c) 2026 iisacc", korean)
        agent = (ROOT / "AGENT.md").read_text(encoding="utf-8")
        self.assertIn("Apache License 2.0", agent)
        self.assertIn("Copyright (c) 2026 iisacc", agent)

    def test_first_party_dependency_statement_is_apache(self):
        text = (ROOT / "docs" / "dependencies.md").read_text(encoding="utf-8")
        self.assertIn("Apache License 2.0(`SPDX-License-Identifier: Apache-2.0`)", text)
        self.assertNotIn("SPDX-License-Identifier: MIT", text)

    def test_package_metadata_exports_spdx_license(self):
        lists = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        config = (ROOT / "cmake" / "iiLocalDiffusionConfig.cmake.in").read_text(encoding="utf-8")
        export = 'set(iiLocalDiffusion_SPDX_LICENSE "Apache-2.0")'
        self.assertIn(export, lists)
        self.assertIn(export, config)
        self.assertIn("install(FILES LICENSE NOTICE README.md THIRD_PARTY_NOTICES.md", lists)


if __name__ == "__main__":
    unittest.main()
