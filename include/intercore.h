#ifndef AS_INTERCORE_H_
#define AS_INTERCORE_H_

#include <applibs/eventloop.h>

#include "exitcode.h"

#define InterCoreRXBuffSize     490
#define InterCoreChunkSize      256

extern float InterCoreRXBuff[InterCoreRXBuffSize];
extern volatile uint32_t InterCoreRXIndex;
extern volatile int intercore_counter;

ExitCode InterCoreInit(EventLoop* event_loop, EventRegistration* socket_event_reg, 
                        int app_socket, const char* rtAppCompID);
ExitCode InterCoreSocketEventHandler(EventLoop* el, int fd, EventLoop_IoEvents events, void* context);
ExitCode InterCoreSendMessage(int socketFd, char* txMessage);

#endif  /* AS_INTERCORE_H_ */