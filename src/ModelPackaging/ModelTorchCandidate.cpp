#include "ModelTorchCandidate.hpp"

#include <algorithm>
#include <array>
#include <fstream>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace iild::detail {
namespace {
using Bytes=std::vector<unsigned char>;
using U64=std::uint64_t;
constexpr U64 maxCentralBytes=64 * 1024 * 1024;
constexpr U64 maxEntries=1000000;

U64 number(const unsigned char *data,std::size_t size) {
    U64 value=0;
    for (std::size_t i=0;i<size;++i) value|=U64(data[i])<<(8*i);
    return value;
}
bool readAt(std::ifstream &file,U64 total,U64 offset,unsigned char *buffer,std::size_t count) {
    if (offset>total || count>total-offset || offset>static_cast<U64>(std::numeric_limits<std::streamoff>::max())) return false;
    file.clear();file.seekg(static_cast<std::streamoff>(offset));
    return static_cast<bool>(file.read(reinterpret_cast<char *>(buffer),static_cast<std::streamsize>(count)));
}
bool dataPkl(std::string_view name) { return name=="data.pkl" || name.ends_with("/data.pkl"); }

// torch.save's legacy stream begins with a pickled constant. Validate that
// constant without interpreting the rest of the pickle or executing it.
bool legacyPickle(std::ifstream &file,U64 total) {
    constexpr std::string_view asciiMagic="L119547037146038801333356L\n.";
    constexpr std::array<unsigned char,13> binaryMagic{
        0x8a,0x0a,0x6c,0xfc,0x9c,0x46,0xf9,0x20,0x6a,0xa8,0x50,0x19,0x2e};
    std::array<unsigned char,28> prefix{};
    const auto length=static_cast<std::size_t>(std::min<U64>(total,prefix.size()));
    if (length<2 || !readAt(file,total,0,prefix.data(),length)) return false;
    if (length>=asciiMagic.size() && std::equal(asciiMagic.begin(),asciiMagic.end(),prefix.begin())) return true;
    if (prefix[0]!=0x80) return false;
    if ((prefix[1]==2 || prefix[1]==3) && length>=15)
        return std::equal(binaryMagic.begin(),binaryMagic.end(),prefix.begin()+2);
    if ((prefix[1]==4 || prefix[1]==5) && length>=24 && prefix[2]==0x95 && number(prefix.data()+3,8)==13)
        return std::equal(binaryMagic.begin(),binaryMagic.end(),prefix.begin()+11);
    return false;
}

bool localPickle(std::ifstream &file,U64 total) {
    std::array<unsigned char,30> header{};
    if (!readAt(file,total,0,header.data(),header.size()) || number(header.data(),4)!=0x04034b50) return false;
    const auto length=number(header.data()+26,2);
    if (!length || length>4096) return false;
    std::string name(static_cast<std::size_t>(length),'\0');
    if (!readAt(file,total,header.size(),reinterpret_cast<unsigned char *>(name.data()),name.size())) return false;
    return dataPkl(name);
}

// A ZIP may place data.pkl after other entries. Read only the bounded central
// directory and support ZIP64 offsets; never decode a payload or pickle.
bool centralPickle(std::ifstream &file,U64 total) {
    constexpr std::size_t eocdSize=22,locatorSize=20,zip64Size=56,centralHeader=46;
    if (total<eocdSize) return false;
    const auto tailSize=static_cast<std::size_t>(std::min<U64>(total,65535+eocdSize));
    Bytes tail(tailSize);
    if (!readAt(file,total,total-tailSize,tail.data(),tail.size())) return false;
    std::size_t eocd=tail.size();
    for (std::size_t i=tail.size()-eocdSize+1;i-->0;) {
        if (number(tail.data()+i,4)==0x06054b50 && i+eocdSize+number(tail.data()+i+20,2)==tail.size()) {eocd=i;break;}
    }
    if (eocd==tail.size()) return false;
    const auto *end=tail.data()+eocd;
    if (number(end+4,2)!=0 || number(end+6,2)!=0) return false;
    U64 diskCount=number(end+8,2),count=number(end+10,2),bytes=number(end+12,4),offset=number(end+16,4);
    const U64 endOffset=total-tailSize+eocd;
    if (diskCount==0xffff || count==0xffff || bytes==0xffffffff || offset==0xffffffff) {
        if (endOffset<locatorSize) return false;
        std::array<unsigned char,locatorSize> locator{};
        if (!readAt(file,total,endOffset-locatorSize,locator.data(),locator.size())
            || number(locator.data(),4)!=0x07064b50 || number(locator.data()+4,4)!=0
            || number(locator.data()+16,4)!=1) return false;
        std::array<unsigned char,zip64Size> extended{};
        if (!readAt(file,total,number(locator.data()+8,8),extended.data(),extended.size())
            || number(extended.data(),4)!=0x06064b50 || number(extended.data()+4,8)<44
            || number(extended.data()+16,4)!=0 || number(extended.data()+20,4)!=0) return false;
        diskCount=number(extended.data()+24,8);count=number(extended.data()+32,8);
        bytes=number(extended.data()+40,8);offset=number(extended.data()+48,8);
    }
    if (diskCount!=count || !count || count>maxEntries || bytes>maxCentralBytes || offset>endOffset || bytes>endOffset-offset) return false;
    U64 cursor=offset;
    const U64 limit=offset+bytes;
    if (offset>static_cast<U64>(std::numeric_limits<std::streamoff>::max())) return false;
    file.clear();file.seekg(static_cast<std::streamoff>(offset));
    if (!file) return false;
    for (U64 i=0;i<count;++i) {
        std::array<unsigned char,centralHeader> header{};
        if (cursor>limit || centralHeader>limit-cursor
            || !file.read(reinterpret_cast<char *>(header.data()),static_cast<std::streamsize>(header.size()))
            || number(header.data(),4)!=0x02014b50) return false;
        const auto nameSize=number(header.data()+28,2),extra=number(header.data()+30,2),comment=number(header.data()+32,2);
        cursor+=centralHeader;
        if (nameSize+extra+comment>limit-cursor) return false;
        if (nameSize && nameSize<=4096) {
            std::string name(static_cast<std::size_t>(nameSize),'\0');
            if (!file.read(name.data(),static_cast<std::streamsize>(name.size()))) return false;
            if (dataPkl(name)) return true;
        } else if (!file.ignore(static_cast<std::streamsize>(nameSize))) return false;
        if (!file.ignore(static_cast<std::streamsize>(extra+comment))) return false;
        cursor+=nameSize+extra+comment;
    }
    return false;
}
}

bool isPyTorchCandidate(const std::filesystem::path &path,std::uint64_t size) {
    std::ifstream file(path,std::ios::binary);
    if (!file) return false;
    std::array<unsigned char,4> prefix{};
    if (size<2 || !readAt(file,size,0,prefix.data(),static_cast<std::size_t>(std::min<U64>(size,prefix.size())))) return false;
    if ((prefix[0]==0x80 || prefix[0]=='L') && legacyPickle(file,size)) return true;
    if (size<4 || number(prefix.data(),4)!=0x04034b50) return false;
    return localPickle(file,size) || centralPickle(file,size);
}
}
