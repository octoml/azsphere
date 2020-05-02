#ifndef AS_NETWORK_H_
#define AS_NETWORK_H_

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
        #if AS_DEBUG
        fprintf(stderr, "ERROR: Networking_IpConfig_Apply: %d (%s)\n", errno, strerror(errno));
        #endif  /* AS_DEBUG */
        return ExitCode_ConfigureStaticIp_IpConfigApply;
    }
    #if AS_DEBUG
    fprintf(stderr, "INFO: Set static IP address on network interface: %s.\n", interfaceName);
    #endif  /* AS_DEBUG */
    
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
    #if AS_DEBUG
    fprintf(stderr, "ERROR: TCP server: \"%s\", errno=%d (%s)\n", desc, errno, strerror(errno));
    #endif  /* AS_DEBUG */
}

ExitCode NetworkEnable(char * interface) {
    // Ensure the necessary network interface is enabled.
    int result = Networking_SetInterfaceState(interface, true);
    if (result != 0) {
        if (errno == EAGAIN) {
            #if AS_DEBUG
            fprintf(stderr, "INFO: The networking stack isn't ready yet, will try again later.\n");
            #endif  /* AS_DEBUG */
            return ExitCode_Success;
        } else {
            #if AS_DEBUG
            fprintf(stderr, 
                "ERROR: Networking_SetInterfaceState for interface '%s' failed: errno=%d (%s)\n",
                interface, errno, strerror(errno));
            #endif  /* AS_DEBUG */
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
            #if AS_DEBUG
            fprintf(stderr, "INFO: The networking stack isn't ready yet, will try again later.\n");
            #endif  /* AS_DEBUG */
            return ExitCode_Success;
        } else {
            #if AS_DEBUG
            fprintf(stderr, 
                "ERROR: Networking_SetInterfaceState for interface '%s' failed: errno=%d (%s)\n",
                interface, errno, strerror(errno));
            #endif  /* AS_DEBUG */
            return ExitCode_CheckStatus_SetInterfaceState;
        }
    }
    // isNetworkStackReady = true;

    // Display total number of network interfaces.
    ssize_t count = Networking_GetInterfaceCount();
    if (count == -1) {
        #if AS_DEBUG
        fprintf(stderr, "ERROR: Networking_GetInterfaceCount: errno=%d (%s)\n", errno, strerror(errno));
        #endif  /* AS_DEBUG */
        return ExitCode_CheckStatus_GetInterfaceCount;
    }
    #if AS_DEBUG
    fprintf(stderr, "INFO: Networking_GetInterfaceCount: count=%zd\n", count);
    #endif  /* AS_DEBUG */
    
    // Read current status of all interfaces.
    size_t bytesRequired = ((size_t)count) * sizeof(Networking_NetworkInterface);
    Networking_NetworkInterface *interfaces = malloc(bytesRequired);
    if (!interfaces) {
        abort();
    }

    ssize_t actualCount = Networking_GetInterfaces(interfaces, (size_t)count);
    if (actualCount == -1) {
        #if AS_DEBUG
        fprintf(stderr, "ERROR: Networking_GetInterfaces: errno=%d (%s)\n", errno, strerror(errno));
        #endif  /* AS_DEBUG */  
    }
    #if AS_DEBUG
    fprintf(stderr, "INFO: Networking_GetInterfaces: actualCount=%zd\n", actualCount);
    #endif  /* AS_DEBUG */
    
    // Print detailed description of each interface.
    #if AS_DEBUG
    for (ssize_t i = 0; i < actualCount; ++i) {
        fprintf(stderr, "INFO: interface #%zd\n", i);

        // Print the interface's name.
        fprintf(stderr, "INFO:   interfaceName=\"%s\"\n", interfaces[i].interfaceName);

        // Print whether the interface is enabled.
        fprintf(stderr, "INFO:   isEnabled=\"%d\"\n", interfaces[i].isEnabled);

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
        fprintf(stderr, "INFO:   ipConfigurationType=%d (%s)\n", ipType, typeText);

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
        fprintf(stderr, "INFO:   interfaceMediumType=%d (%s)\n", mediumType, mediumText);

        // Print the interface connection status
        Networking_InterfaceConnectionStatus status;
        int result = Networking_GetInterfaceConnectionStatus(interfaces[i].interfaceName, &status);
        if (result != 0) {
            fprintf(stderr, "ERROR: Networking_GetInterfaceConnectionStatus: errno=%d (%s)\n", errno,
                      strerror(errno));
            return ExitCode_CheckStatus_GetInterfaceConnectionStatus;
        }
        fprintf(stderr, "INFO:   interfaceStatus=0x%02x\n", status);
    }
    #endif  /* AS_DEBUG */

    free(interfaces);

    return ExitCode_Success;
}

int message(uint16_t id, Message type, char * message) {
    int len = 0;
    message[0] = ((unsigned char)id >> 8) & 0xFF;
    message[1] = (unsigned char)id & 0xFF;
    len += 2;

    switch (type)
    {
    case Message_START:
        message[len] = ',';
        message[len+1] = 'S';
        message[len+2] = 'T';
        message[len+3] = 'A';
        message[len+4] = 'R';
        message[len+5] = 'T';
        message[len+6] = '\n';
        len += 7;
        break;
    case Message_TIME:
        message[len] = ',';
        message[len+1] = 'T';
        message[len+2] = 'I';
        message[len+3] = 'M';
        message[len+4] = 'E';
        len += 5;
        break;
    case Message_RESULT:
        message[len] = ',';
        message[len+1] = 'R';
        message[len+2] = 'E';
        message[len+3] = 'S';
        len += 4;
        break;
    default:
        break;
    }
    return len;
}

#endif  /* AS_NETWORK_H_ */