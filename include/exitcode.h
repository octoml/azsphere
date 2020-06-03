/* Copyright (c) Microsoft Corporation. All rights reserved.
   Licensed under the MIT License. */

#pragma once

typedef enum {
  ExitCode_Success =                                              0,
  ExitCode_TermHandler_SigTerm =                                  1,
  ExitCode_StoppedHandler_Stopped =                               2,
  ExitCode_CheckStatus_SetInterfaceState =                        3,
  ExitCode_CheckStatus_GetInterfaceCount =                        4,
  ExitCode_CheckStatus_GetInterfaceConnectionStatus =             5,
  ExitCode_ConfigureStaticIp_IpConfigApply =                      6,
  ExitCode_StartSntpServer_StartSntp =                            7,
  ExitCode_StartDhcpServer_StartDhcp =                            8,
  ExitCode_TimerHandler_Consume =                                 9,
  ExitCode_InitLaunch_Epoll =                                     10,
  ExitCode_InitLaunch_Timer =                                     11,
  ExitCode_Main_WaitCallFailure =                                 12,
  ExitCode_EchoStart_Listen =                                     13,
  ExitCode_OpenIpV4_Socket =                                      14,
  ExitCode_OpenIpV4_SetSockOpt =                                  15,
  ExitCode_OpenIpV4_Bind =                                        16,

  ExitCode_Main_Led =                                             17,
  ExitCode_TermHandler_SigKill =                                  18,

  ExitCode_SendMsg_Send =                                         19,
  ExitCode_SocketHandler_Recv =                                   20,
  ExitCode_Init_EventLoop =                                       21,
  ExitCode_Init_SendTimer =                                       22,
  ExitCode_Init_Connection =                                      23,
  ExitCode_Init_SetSockOpt =                                      24,
  ExitCode_Init_RegisterIo =                                      25,
  ExitCode_Main_EventLoopFail =                                   26,

  ExitCode_TVM_Init =                                             27
} ExitCode;


    
    