#ifndef AS_UTILS_H_
#define AS_UTILS_H_

#include <stdio.h>

#define AS_DEBUG    0

int ReadID(const char * filename, uint16_t * id) {
  size_t fs;
  int fid;

  fid = Storage_OpenFileInImagePackage(filename);
  if (fid == -1) {
      #if AS_DEBUG
      fprintf(stderr, "Error: Openning %s failed!\n", filename);
      #endif  /* AS_DEBUG */
      return -1;
  }
  fs = (size_t)lseek(fid, 0, SEEK_END);
  if (fs == -1) {
      #if AS_DEBUG
      fprintf(stderr, "Error: File %s size!\n", filename);
      #endif  /* AS_DEBUG */
      return -1;
  }
  lseek(fid, 0, SEEK_SET);
  read(fid, id, sizeof(id));
  close(fid);
  return 0;
}

int Read_File_Char(const char * filename, char** data) {
  size_t fs;
  int fid;
  *data = NULL;

  fid = Storage_OpenFileInImagePackage(filename);
  if (fid == -1) {
    #if AS_DEBUG
    fprintf(stderr, "Error: Openning %s failed!\n", filename);
    #endif  /* AS_DEBUG */
    return -1;
  }
  fs = (size_t)lseek(fid, 0, SEEK_END);
  if (fs == -1) {
    #if AS_DEBUG
    fprintf(stderr, "Error: File %s size!\n", filename);
    #endif  /* AS_DEBUG */
    return -1;
  }
  *data = (char*)malloc(fs);
  if (*data == NULL) {
    #if AS_DEBUG
    fprintf(stderr, "Unable to allocate memory");
    #endif  /* AS_DEBUG */
    return -1;
  }
  lseek(fid, 0, SEEK_SET);
  #if AS_DEBUG
  fprintf(stderr, "%s size: %d\n", filename, (int)fs);
  #endif  /* AS_DEBUG */
  read(fid, *data, fs);
  close(fid);
  return (int) fs;
}

int Read_File_Float(const char * filename, float** data) {
  int fid;
  size_t fs;
  *data = NULL;

  fid = Storage_OpenFileInImagePackage(filename);
  if (fid == -1) {
    #if AS_DEBUG
    fprintf(stderr, "Error: Openning %s failed!\n", filename);
    #endif  /* AS_DEBUG */
    return -1;
  }
  fs = (size_t)lseek(fid, 0, SEEK_END);
  if (fs == -1) {
    #if AS_DEBUG
    fprintf(stderr, "Error: File %s size!\n", filename);
    #endif  /* AS_DEBUG */
    return -1;
  }
  lseek(fid, 0, SEEK_SET);
  *data = (float *)malloc(fs);
  if (*data == NULL) {
    #if AS_DEBUG
    fprintf(stderr, "Unable to allocate memory");
    #endif  /* AS_DEBUG */
    return -1;
  }
  #if AS_DEBUG
  fprintf(stderr, "%s size: %d\n", data_file, (int)fs);
  #endif  /* AS_DEBUG */
  read(fid, *data, fs);
  close(fid);
}











#endif  /* AS_UTILS_H_ */