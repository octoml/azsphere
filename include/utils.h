#ifndef AS_UTILS_H_
#define AS_UTILS_H_

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <stdint.h>

#define DEBUG_MESSAGE_MAX_LENGTH  128

#if AS_NETWORK_DEBUG
int Log_Debug(const char *fmt, ...);
#endif

void Debug_Init(int socket);
int ReadID(const char * filename, uint16_t * id);
int Read_File_Int8(const char * filename, void** data);
int Read_File_Int32(const char * filename, void** data);
int Read_File_Char(const char * filename, char** data);
int Read_File_Float(const char * filename, float** data);
void CloseFdAndPrintError(int fd, const char *fdName);

#endif  /* AS_UTILS_H_ */