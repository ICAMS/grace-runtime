//Copyright (C) 2011  Carl Rogers
//Released under MIT License
//license available in LICENSE file, or at http://www.opensource.org/licenses/mit-license.php

#include"cnpy.h"
#include<complex>
#include<cstdlib>
#include<algorithm>
#include<cstring>
#include<iomanip>
#include<stdint.h>
#include<stdexcept>
#include <regex>

namespace grace {

char cnpy::BigEndianTest() {
    int x = 1;
    return (((char *)&x)[0]) ? '<' : '>';
}

char cnpy::map_type(const std::type_info& t)
{
    if(t == typeid(float) ) return 'f';
    if(t == typeid(double) ) return 'f';
    if(t == typeid(long double) ) return 'f';

    if(t == typeid(int) ) return 'i';
    if(t == typeid(char) ) return 'i';
    if(t == typeid(short) ) return 'i';
    if(t == typeid(long) ) return 'i';
    if(t == typeid(long long) ) return 'i';

    if(t == typeid(unsigned char) ) return 'u';
    if(t == typeid(unsigned short) ) return 'u';
    if(t == typeid(unsigned long) ) return 'u';
    if(t == typeid(unsigned long long) ) return 'u';
    if(t == typeid(unsigned int) ) return 'u';

    if(t == typeid(bool) ) return 'b';

    if(t == typeid(std::complex<float>) ) return 'c';
    if(t == typeid(std::complex<double>) ) return 'c';
    if(t == typeid(std::complex<long double>) ) return 'c';

    else return '?';
}

template<> std::vector<char>& cnpy::operator+=(std::vector<char>& lhs, const std::string rhs) {
    lhs.insert(lhs.end(),rhs.begin(),rhs.end());
    return lhs;
}

template<> std::vector<char>& cnpy::operator+=(std::vector<char>& lhs, const char* rhs) {
    //write in little endian
    size_t len = strlen(rhs);
    lhs.reserve(len);
    for(size_t byte = 0; byte < len; byte++) {
        lhs.push_back(rhs[byte]);
    }
    return lhs;
}

void cnpy::parse_npy_header(unsigned char* buffer,size_t& word_size, std::vector<size_t>& shape, bool& fortran_order) {
    //std::string magic_string(buffer,6);
    uint8_t major_version = *reinterpret_cast<uint8_t*>(buffer+6);
    uint8_t minor_version = *reinterpret_cast<uint8_t*>(buffer+7);
    uint16_t header_len = *reinterpret_cast<uint16_t*>(buffer+8);
    std::string header(reinterpret_cast<char*>(buffer+9),header_len);

    size_t loc1, loc2;

    //fortran order
    loc1 = header.find("fortran_order")+16;
    fortran_order = (header.substr(loc1,4) == "True" ? true : false);

    //shape
    loc1 = header.find("(");
    loc2 = header.find(")");

    std::regex num_regex("[0-9][0-9]*");
    std::smatch sm;
    shape.clear();

    std::string str_shape = header.substr(loc1+1,loc2-loc1-1);
    while(std::regex_search(str_shape, sm, num_regex)) {
        shape.push_back(std::stoi(sm[0].str()));
        str_shape = sm.suffix().str();
    }

    //endian, word size, data type
    //byte order code | stands for not applicable. 
    //not sure when this applies except for byte array
    loc1 = header.find("descr")+9;
    bool littleEndian = (header[loc1] == '<' || header[loc1] == '|' ? true : false);
    assert(littleEndian);

    //char type = header[loc1+1];
    //assert(type == map_type(T));

    std::string str_ws = header.substr(loc1+2);
    loc2 = str_ws.find("'");
    word_size = atoi(str_ws.substr(0,loc2).c_str());
}

// grace: the header length of format 1.0 (2 bytes) and 2.0 / 3.0 (4 bytes) instead of one fgets of at most 256 bytes;
// the dtype descr is kept, and dtypes this reader would misread (big-endian, structured, object) are rejected instead
// of being read as native little-endian data of their word size
void cnpy::parse_npy_header(FILE* fp, size_t& word_size, std::vector<size_t>& shape, bool& fortran_order,
                            std::string& descr) {
    unsigned char pre[10];
    if(fread(pre,1,10,fp) != 10)
        throw std::runtime_error("parse_npy_header: failed fread");
    if(pre[0] != 0x93 || std::memcmp(pre+1,"NUMPY",5) != 0)
        throw std::runtime_error("parse_npy_header: not a .npy record");
    size_t header_len = (size_t) pre[8] | ((size_t) pre[9] << 8);
    if(pre[6] >= 2) {
        unsigned char ext[2];
        if(fread(ext,1,2,fp) != 2)
            throw std::runtime_error("parse_npy_header: failed fread");
        header_len |= ((size_t) ext[0] << 16) | ((size_t) ext[1] << 24);
    }
    std::string header(header_len,' ');
    if(header_len == 0 || fread(&header[0],1,header_len,fp) != header_len)
        throw std::runtime_error("parse_npy_header: failed fread");

    size_t loc1, loc2;

    //fortran order
    loc1 = header.find("fortran_order");
    if (loc1 == std::string::npos)
        throw std::runtime_error("parse_npy_header: failed to find header keyword: 'fortran_order'");
    loc1 += 16;
    fortran_order = (header.substr(loc1,4) == "True" ? true : false);

    //shape
    loc1 = header.find("(");
    loc2 = header.find(")");
    if (loc1 == std::string::npos || loc2 == std::string::npos)
        throw std::runtime_error("parse_npy_header: failed to find header keyword: '(' or ')'");

    std::regex num_regex("[0-9][0-9]*");
    std::smatch sm;
    shape.clear();

    std::string str_shape = header.substr(loc1+1,loc2-loc1-1);
    while(std::regex_search(str_shape, sm, num_regex)) {
        shape.push_back(std::stoull(sm[0].str()));
        str_shape = sm.suffix().str();
    }

    //dtype: '<f8', '|S2', ...; byte order '|': not applicable (one-byte types)
    loc1 = header.find("descr");
    if (loc1 == std::string::npos)
        throw std::runtime_error("parse_npy_header: failed to find header keyword: 'descr'");
    loc1 = header.find(':', loc1);
    loc1 = loc1 == std::string::npos ? loc1 : header.find_first_not_of(' ', loc1 + 1);
    if (loc1 == std::string::npos || header[loc1] != '\'')
        throw std::runtime_error("parse_npy_header: structured dtypes are not supported");
    loc2 = header.find('\'', loc1 + 1);
    if (loc2 == std::string::npos)
        throw std::runtime_error("parse_npy_header: malformed 'descr'");
    descr = header.substr(loc1 + 1, loc2 - loc1 - 1);
    if (descr.size() < 3 || std::string("<>|=").find(descr[0]) == std::string::npos)
        throw std::runtime_error("parse_npy_header: unsupported dtype '" + descr + "'");
    word_size = (size_t) atoi(descr.substr(2).c_str());
    if (descr[1] == 'U') word_size *= 4;    // numpy unicode: the size counts UCS-4 characters
    if (word_size == 0)
        throw std::runtime_error("parse_npy_header: unsupported dtype '" + descr + "'");
    if (word_size > 1 && descr[0] == (BigEndianTest() == '<' ? '>' : '<'))
        throw std::runtime_error("parse_npy_header: dtype '" + descr + "' is not in the byte order of this machine");
}

void cnpy::parse_npy_header(FILE* fp, size_t& word_size, std::vector<size_t>& shape, bool& fortran_order) {
    std::string descr;    // npy_save (append mode)
    parse_npy_header(fp, word_size, shape, fortran_order, descr);
}

void cnpy::parse_zip_footer(FILE* fp, uint16_t& nrecs, size_t& global_header_size, size_t& global_header_offset)
{
    std::vector<char> footer(22);
    fseek(fp,-22,SEEK_END);
    size_t res = fread(&footer[0],sizeof(char),22,fp);
    if(res != 22)
        throw std::runtime_error("parse_zip_footer: failed fread");

    uint16_t disk_no, disk_start, nrecs_on_disk, comment_len;
    disk_no = *(uint16_t*) &footer[4];
    disk_start = *(uint16_t*) &footer[6];
    nrecs_on_disk = *(uint16_t*) &footer[8];
    nrecs = *(uint16_t*) &footer[10];
    global_header_size = *(uint32_t*) &footer[12];
    global_header_offset = *(uint32_t*) &footer[16];
    comment_len = *(uint16_t*) &footer[20];

    assert(disk_no == 0);
    assert(disk_start == 0);
    assert(nrecs_on_disk == nrecs);
    assert(comment_len == 0);
}

cnpy::NpyArray load_the_npy_file(FILE* fp) {
    std::vector<size_t> shape;
    size_t word_size;
    bool fortran_order;
    std::string descr;
    cnpy::parse_npy_header(fp,word_size,shape,fortran_order,descr);

    cnpy::NpyArray arr(shape, word_size, fortran_order);
    arr.descr = descr;
    size_t nread = fread(arr.data<char>(),1,arr.num_bytes(),fp);
    if(nread != arr.num_bytes())
        throw std::runtime_error("load_the_npy_file: failed fread");
    return arr;
}

//cnpy::NpyArray load_the_npz_array(FILE* fp, uint32_t compr_bytes, uint32_t uncompr_bytes) {
//
//    std::vector<unsigned char> buffer_compr(compr_bytes);
//    std::vector<unsigned char> buffer_uncompr(uncompr_bytes);
//    size_t nread = fread(&buffer_compr[0],1,compr_bytes,fp);
//    if(nread != compr_bytes)
//        throw std::runtime_error("load_the_npy_file: failed fread");
//
//    int err;
//    z_stream d_stream;
//
//    d_stream.zalloc = Z_NULL;
//    d_stream.zfree = Z_NULL;
//    d_stream.opaque = Z_NULL;
//    d_stream.avail_in = 0;
//    d_stream.next_in = Z_NULL;
//    err = inflateInit2(&d_stream, -MAX_WBITS);
//
//    d_stream.avail_in = compr_bytes;
//    d_stream.next_in = &buffer_compr[0];
//    d_stream.avail_out = uncompr_bytes;
//    d_stream.next_out = &buffer_uncompr[0];
//
//    err = inflate(&d_stream, Z_FINISH);
//    err = inflateEnd(&d_stream);
//
//    std::vector<size_t> shape;
//    size_t word_size;
//    bool fortran_order;
//    cnpy::parse_npy_header(&buffer_uncompr[0],word_size,shape,fortran_order);
//
//    cnpy::NpyArray array(shape, word_size, fortran_order);
//
//    size_t offset = uncompr_bytes - array.num_bytes();
//    memcpy(array.data<unsigned char>(),&buffer_uncompr[0]+offset,array.num_bytes());
//
//    return array;
//}

cnpy::npz_t cnpy::npz_load(std::string fname) {
    FILE* fp = fopen(fname.c_str(),"rb");

    if(!fp) {
        throw std::runtime_error("npz_load: Error! Unable to open file "+fname+"!");
    }
    // grace: closed on every exit, the errors below included
    std::unique_ptr<FILE, int (*)(FILE *)> fp_guard(fp, fclose);

    cnpy::npz_t arrays;  

    while(1) {
        std::vector<char> local_header(30);
        size_t headerres = fread(&local_header[0],sizeof(char),30,fp);
        if(headerres != 30)
            throw std::runtime_error("npz_load: failed fread");

        //if we've reached the global header, stop reading
        if(local_header[2] != 0x03 || local_header[3] != 0x04) break;

        //read in the variable name
        uint16_t name_len = *(uint16_t*) &local_header[26];
        std::string varname(name_len,' ');
        size_t vname_res = fread(&varname[0],sizeof(char),name_len,fp);
        if(vname_res != name_len)
            throw std::runtime_error("npz_load: failed fread");

        //erase the lagging .npy        
        varname.erase(varname.end()-4,varname.end());

        //read in the extra field
        uint16_t extra_field_len = *(uint16_t*) &local_header[28];
        if(extra_field_len > 0) {
            std::vector<char> buff(extra_field_len);
            size_t efield_res = fread(&buff[0],sizeof(char),extra_field_len,fp);
            if(efield_res != extra_field_len)
                throw std::runtime_error("npz_load: failed fread");
        }

        uint16_t compr_method = *reinterpret_cast<uint16_t*>(&local_header[0]+8);
        uint32_t compr_bytes = *reinterpret_cast<uint32_t*>(&local_header[0]+18);
        uint32_t uncompr_bytes = *reinterpret_cast<uint32_t*>(&local_header[0]+22);

        // grace: no zlib here; a compressed member used to end the loop silently with the remaining arrays missing
        if(compr_method != 0)
            throw std::runtime_error("npz_load: '" + varname + "' in " + fname +
                                     " is compressed (np.savez_compressed); save the file with np.savez");
        arrays[varname] = load_the_npy_file(fp);
//        else {arrays[varname] = load_the_npz_array(fp,compr_bytes,uncompr_bytes);}
    }

    return arrays;  
}

cnpy::NpyArray cnpy::npz_load(std::string fname, std::string varname) {
    FILE* fp = fopen(fname.c_str(),"rb");

    if(!fp) throw std::runtime_error("npz_load: Unable to open file "+fname);

    while(1) {
        std::vector<char> local_header(30);
        size_t header_res = fread(&local_header[0],sizeof(char),30,fp);
        if(header_res != 30)
            throw std::runtime_error("npz_load: failed fread");

        //if we've reached the global header, stop reading
        if(local_header[2] != 0x03 || local_header[3] != 0x04) break;

        //read in the variable name
        uint16_t name_len = *(uint16_t*) &local_header[26];
        std::string vname(name_len,' ');
        size_t vname_res = fread(&vname[0],sizeof(char),name_len,fp);      
        if(vname_res != name_len)
            throw std::runtime_error("npz_load: failed fread");
        vname.erase(vname.end()-4,vname.end()); //erase the lagging .npy

        //read in the extra field
        uint16_t extra_field_len = *(uint16_t*) &local_header[28];
        fseek(fp,extra_field_len,SEEK_CUR); //skip past the extra field
        
        uint16_t compr_method = *reinterpret_cast<uint16_t*>(&local_header[0]+8);
        uint32_t compr_bytes = *reinterpret_cast<uint32_t*>(&local_header[0]+18);
        uint32_t uncompr_bytes = *reinterpret_cast<uint32_t*>(&local_header[0]+22);

        if(vname == varname) {
            if(compr_method != 0) {
                fclose(fp);
                throw std::runtime_error("npz_load: '" + varname + "' in " + fname +
                                         " is compressed (np.savez_compressed); save the file with np.savez");
            }
//            NpyArray array  = (compr_method == 0) ? load_the_npy_file(fp) : load_the_npz_array(fp,compr_bytes,uncompr_bytes);
            NpyArray array  =  load_the_npy_file(fp);
            fclose(fp);
            return array;
        }
        else {
            //skip past the data
            uint32_t size = *(uint32_t*) &local_header[22];
            fseek(fp,size,SEEK_CUR);
        }
    }

    fclose(fp);

    //if we get here, we haven't found the variable in the file
    throw std::runtime_error("npz_load: Variable name "+varname+" not found in "+fname);
}

cnpy::NpyArray cnpy::npy_load(std::string fname) {

    FILE* fp = fopen(fname.c_str(), "rb");

    if(!fp) throw std::runtime_error("npy_load: Unable to open file "+fname);

    NpyArray arr = load_the_npy_file(fp);

    fclose(fp);
    return arr;
}




}  // namespace grace
