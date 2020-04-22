#ifndef NETWORK_H_
#define NETWORK_H_

#include <applibs/networking.h>
#include <sys/socket.h>

#include "exitcode.h"


typedef enum {
    Message_START       = 0,
    Message_TIME        = 1,
    Message_RESULT      = 2,
} Message;

static void ReportError(const char *desc);


static ExitCode ConfigureNetworkInterfaceWithStaticIp(const char *interfaceName,
                                                      char * ipAddress,
                                                      char * subnet,
                                                      char * gateway 
                                                     )
{
    Networking_IpConfig ipConfig;
    struct in_addr localIPAddress;
    struct in_addr subnetMask;
    struct in_addr gatewayIPAddress;

    Networking_IpConfig_Init(&ipConfig);
    inet_aton(ipAddress, &localIPAddress);
    inet_aton(subnet, &subnetMask);
    inet_aton(gateway, &gatewayIPAddress);

    Networking_IpConfig_EnableStaticIp(&ipConfig, localIPAddress, subnetMask,
                                       gatewayIPAddress);

    int r = Networking_IpConfig_Apply(interfaceName, &ipConfig);
    Networking_IpConfig_Destroy(&ipConfig);
    if (r != 0) {
        Log_Debug("ERROR: Networking_IpConfig_Apply: %d (%s)\n", errno, strerror(errno));
        return ExitCode_ConfigureStaticIp_IpConfigApply;
    }
    Log_Debug("INFO: Set static IP address on network interface: %s.\n", interfaceName);

    return ExitCode_Success;
}

int OpenIpV4Socket(char * ip, uint16_t port, int sockType, ExitCode *callerExitCode)
{
    int localFd = -1;
    int retFd = -1;

    do {
        // Create a TCP / IPv4 socket.
        localFd = socket(AF_INET, sockType, /* protocol */ 0);
        if (localFd < 0) {
            ReportError("socket");
            *callerExitCode = ExitCode_OpenIpV4_Socket;
            break;
        }

        struct sockaddr_in serv_addr; 
        serv_addr.sin_family = AF_INET; 
        serv_addr.sin_port = htons(port);
        serv_addr.sin_addr.s_addr = inet_addr(ip);
        int r = connect(localFd, (struct sockaddr *)&serv_addr, sizeof(serv_addr));

        if (r != 0) {
            ReportError("connect");
            *callerExitCode = ExitCode_OpenIpV4_SetSockOpt;
            break;
        }

        retFd = localFd;
        localFd = -1;
    } while (0);

    close(localFd);

    return retFd;
}

static void ReportError(const char *desc)
{
    Log_Debug("ERROR: TCP server: \"%s\", errno=%d (%s)\n", desc, errno, strerror(errno));
}

ExitCode NetworkEnable(char * interface) {
    // Ensure the necessary network interface is enabled.
    int result = Networking_SetInterfaceState(interface, true);
    if (result != 0) {
        if (errno == EAGAIN) {
            Log_Debug("INFO: The networking stack isn't ready yet, will try again later.\n");
            return ExitCode_Success;
        } else {
            Log_Debug(
                "ERROR: Networking_SetInterfaceState for interface '%s' failed: errno=%d (%s)\n",
                interface, errno, strerror(errno));
            return ExitCode_CheckStatus_SetInterfaceState;
        }
    }
    return ExitCode_Success;
}

static ExitCode CheckNetworkStatus(char * interface)
{
    // Ensure the necessary network interface is enabled.
    int result = Networking_SetInterfaceState(interface, true);
    if (result != 0) {
        if (errno == EAGAIN) {
            Log_Debug("INFO: The networking stack isn't ready yet, will try again later.\n");
            return ExitCode_Success;
        } else {
            Log_Debug(
                "ERROR: Networking_SetInterfaceState for interface '%s' failed: errno=%d (%s)\n",
                interface, errno, strerror(errno));
            return ExitCode_CheckStatus_SetInterfaceState;
        }
    }
    // isNetworkStackReady = true;

    // Display total number of network interfaces.
    ssize_t count = Networking_GetInterfaceCount();
    if (count == -1) {
        Log_Debug("ERROR: Networking_GetInterfaceCount: errno=%d (%s)\n", errno, strerror(errno));
        return ExitCode_CheckStatus_GetInterfaceCount;
    }
    Log_Debug("INFO: Networking_GetInterfaceCount: count=%zd\n", count);

    // Read current status of all interfaces.
    size_t bytesRequired = ((size_t)count) * sizeof(Networking_NetworkInterface);
    Networking_NetworkInterface *interfaces = malloc(bytesRequired);
    if (!interfaces) {
        abort();
    }

    ssize_t actualCount = Networking_GetInterfaces(interfaces, (size_t)count);
    if (actualCount == -1) {
        Log_Debug("ERROR: Networking_GetInterfaces: errno=%d (%s)\n", errno, strerror(errno));
    }
    Log_Debug("INFO: Networking_GetInterfaces: actualCount=%zd\n", actualCount);

    // Print detailed description of each interface.
    for (ssize_t i = 0; i < actualCount; ++i) {
        Log_Debug("INFO: interface #%zd\n", i);

        // Print the interface's name.
        Log_Debug("INFO:   interfaceName=\"%s\"\n", interfaces[i].interfaceName);

        // Print whether the interface is enabled.
        Log_Debug("INFO:   isEnabled=\"%d\"\n", interfaces[i].isEnabled);

        // Print the interface's configuration type.
        Networking_IpType ipType = interfaces[i].ipConfigurationType;
        const char *typeText;
        switch (ipType) {
        case Networking_IpType_DhcpNone:
            typeText = "DhcpNone";
            break;
        case Networking_IpType_DhcpClient:
            typeText = "DhcpClient";
            break;
        default:
            typeText = "unknown-configuration-type";
            break;
        }
        Log_Debug("INFO:   ipConfigurationType=%d (%s)\n", ipType, typeText);

        // Print the interface's medium.
        Networking_InterfaceMedium_Type mediumType = interfaces[i].interfaceMediumType;
        const char *mediumText;
        switch (mediumType) {
        case Networking_InterfaceMedium_Unspecified:
            mediumText = "unspecified";
            break;
        case Networking_InterfaceMedium_Wifi:
            mediumText = "Wi-Fi";
            break;
        case Networking_InterfaceMedium_Ethernet:
            mediumText = "Ethernet";
            break;
        default:
            mediumText = "unknown-medium";
            break;
        }
        Log_Debug("INFO:   interfaceMediumType=%d (%s)\n", mediumType, mediumText);

        // Print the interface connection status
        Networking_InterfaceConnectionStatus status;
        int result = Networking_GetInterfaceConnectionStatus(interfaces[i].interfaceName, &status);
        if (result != 0) {
            Log_Debug("ERROR: Networking_GetInterfaceConnectionStatus: errno=%d (%s)\n", errno,
                      strerror(errno));
            return ExitCode_CheckStatus_GetInterfaceConnectionStatus;
        }
        Log_Debug("INFO:   interfaceStatus=0x%02x\n", status);
    }

    free(interfaces);

    return ExitCode_Success;
}

int message(char * id, Message type, char * message) {
    int len = 0;
    message[0] = id[0];
    message[1] = id[1];
    message[2] = id[2];
    message[3] = id[3];
    len += 4;

    switch (type)
    {
    case Message_START:
        message[4] = ',';
        message[5] = 'S';
        message[6] = 'T';
        message[7] = 'A';
        message[8] = 'R';
        message[9] = 'T';
        message[10] = '\n';
        len += 7;
        break;
    case Message_TIME:
        message[4] = ',';
        message[5] = 'T';
        message[6] = 'I';
        message[7] = 'M';
        message[8] = 'E';
        len += 5;
        break;
    case Message_RESULT:
        message[4] = ',';
        message[5] = 'R';
        message[6] = 'E';
        message[7] = 'S';
        len += 4;
        break;
    default:
        break;
    }
    return len;
}

#endif  /* NETWORK_H_ */