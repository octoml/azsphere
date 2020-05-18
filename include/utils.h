#ifndef AS_UTILS_H_
#define AS_UTILS_H_

#include <stdio.h>

#define DEBUG_MESSAGE_MAX_LENGTH  128
static int DEBUG_SOCKET = -1;


#if AS_NETWORKING
int fprintf(FILE * file, const char * format, ...)
{
    va_list args;
    va_start(args, format);
    int status;

    if (DEBUG_SOCKET < 0) {
      //standard route
      status = vfprintf(file, format, args);
    }
    else {
      //using Ethernet
      char message [DEBUG_MESSAGE_MAX_LENGTH];
      memset(message, 0, DEBUG_MESSAGE_MAX_LENGTH);
      int status;
      if (message < 0) {
        status = -1;
        return status;
      }
      status = vsnprintf(message, DEBUG_MESSAGE_MAX_LENGTH, format, args);
      send(DEBUG_SOCKET, message, status, 0);
    }
    
    va_end(args);
    return status;
}
#endif

void Debug_Init(int socket) {
  DEBUG_SOCKET = socket;
}

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

int Read_File_Int8(const char * filename, void** data) {
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
  *data = (int8_t*)malloc(fs);
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

int Read_File_Int32(const char * filename, void** data) {
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
  *data = (int32_t*)malloc(fs);
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
  fprintf(stderr, "%s size: %d\n", filename, (int)fs);
  #endif  /* AS_DEBUG */
  read(fid, *data, fs);
  close(fid);
}











#endif  /* AS_UTILS_H_ */