#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <ctype.h>
#include <sys/socket.h>

#include <applibs/eventloop.h>
#include <applibs/application.h>
#include <applibs/log.h>

#include "include/intercore.h"
#include "include/exitcode.h"
#include "include/config.h"

ExitCode InterCoreInit(EventLoop* event_loop, EventRegistration* socket_event_reg, 
                        int app_socket, const char* rtAppCompID)
{
  // Open a connection to the RTApp.
  app_socket = Application_Connect(rtAppCompID);
  if (app_socket == -1) {
    fprintf(stdout, "ERROR: Unable to create socket: %d (%s)\n", errno, strerror(errno));
    return ExitCode_Init_Connection;
  }
  
  // Register handler for incoming messages from real-time capable application.
  socket_event_reg = EventLoop_RegisterIo(event_loop, app_socket, EventLoop_Input, 
                                         InterCoreSocketEventHandler, /* context */ NULL);
  if (socket_event_reg == NULL) {
    fprintf(stdout, "ERROR: Unable to register socket event: %d (%s)\n", errno, strerror(errno));
    return ExitCode_Init_RegisterIo;
  }

  return ExitCode_Success;
}

ExitCode InterCoreSocketEventHandler(EventLoop* el, int fd, EventLoop_IoEvents events, void* context)
{
  char* data = (char *)InterCoreRXBuff;
  int bytesReceived = recv(fd, (data + InterCoreRXIndex), InterCoreChunkSize, 0);

  if (bytesReceived == -1) {
    fprintf(stdout, "ERROR: Unable to receive message: %d (%s)\n", errno, strerror(errno));
    return ExitCode_SocketHandler_Recv;
  }
  InterCoreRXIndex += bytesReceived;
  intercore_counter += 1;
  
  Log_Debug("Received %d bytes: ", bytesReceived);
  return ExitCode_Success;
}

ExitCode InterCoreSendMessage(int socketFd, char* txMessage)
{
  static int iter = 0;

  snprintf(txMessage, sizeof(txMessage), "hl-app-to-rt-app-%02d", iter);
  iter = (iter + 1) % 100;
  fprintf(stdout, "Sending: %s\n", txMessage);

  int bytesSent = send(socketFd, txMessage, strlen(txMessage), 0);
  if (bytesSent == -1) {
    fprintf(stdout, "ERROR: Unable to send message: %d (%s)\n", errno, strerror(errno));
    return ExitCode_SendMsg_Send;
  }
}