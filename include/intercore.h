#ifndef AS_INTERCORE_H_
#define AS_INTERCORE_H_

#include <applibs/eventloop.h>

#include "exitcode.h"

extern char InterCoreRXBuff [32];
extern bool InterCoreRXFlag;

ExitCode InterCoreInit(EventLoop* event_loop, EventRegistration* socket_event_reg, 
                        int app_socket, const char* rtAppCompID);
ExitCode InterCoreSocketEventHandler(EventLoop* el, int fd, EventLoop_IoEvents events, void* context);
ExitCode InterCoreSendMessage(int socketFd, char* txMessage);

#endif  /* AS_INTERCORE_H_ */