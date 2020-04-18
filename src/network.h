#ifndef NETWORK_H_
#define NETWORK_H_

#include <applibs/networking.h>
#include <sys/socket.h>

#include "exitcode.h"

// Ethernet / TCP server settings.
static struct in_addr localServerIpAddress;
static struct in_addr subnetMask;
static struct in_addr gatewayIpAddress;
static const uint16_t LocalTcpServerPort = 11000;
static int serverBacklogSize = 3;
static const char NetworkInterface[] = "eth0";
static bool isNetworkStackReady = false;

static ExitCode ConfigureNetworkInterfaceWithStaticIp(const char *interfaceName);
static int OpenIpV4Socket(in_addr_t ipAddr, uint16_t port, int sockType, ExitCode *callerExitCode);
static void ReportError(const char *desc);
static ExitCode CheckNetworkStatus(void);

static ExitCode ConfigureNetworkInterfaceWithStaticIp(const char *interfaceName)
{
    Networking_IpConfig ipConfig;
    Networking_IpConfig_Init(&ipConfig);
    if (inet_aton("192.168.0.20", &localServerIpAddress) <= 0) {
        ReportError("StaticIP: localServerIpAddress\n");
    }
    if (inet_aton("255.255.255.0", &subnetMask) <= 0) {
        ReportError("StaticIP: subnetMask\n");
    }
    if (inet_aton("192.168.0.1", &gatewayIpAddress) <= 0) {
        ReportError("StaticIP: gatewayIpAddress\n");
    }
    Networking_IpConfig_EnableStaticIp(&ipConfig, localServerIpAddress, subnetMask,
                                       gatewayIpAddress);

    int result = Networking_IpConfig_Apply(interfaceName, &ipConfig);
    Networking_IpConfig_Destroy(&ipConfig);
    if (result != 0) {
        Log_Debug("ERROR: Networking_IpConfig_Apply: %d (%s)\n", errno, strerror(errno));
        return ExitCode_ConfigureStaticIp_IpConfigApply;
    }
    Log_Debug("INFO: Set static IP address on network interface: %s.\n", interfaceName);

    return ExitCode_Success;
}

static int OpenIpV4Socket(in_addr_t ipAddr, uint16_t port, int sockType, ExitCode *callerExitCode)
{
    int localFd = -1;
    int retFd = -1;

    do {
        // Create a TCP / IPv4 socket. This will form the listen socket.
        localFd = socket(AF_INET, SOCK_STREAM, /* protocol */ 0);
        if (localFd < 0) {
            ReportError("socket");
            *callerExitCode = ExitCode_OpenIpV4_Socket;
            break;
        }

        struct sockaddr_in serv_addr; 
        serv_addr.sin_family = AF_INET; 
        serv_addr.sin_port = htons(11000);
        serv_addr.sin_addr.s_addr = inet_addr("192.168.0.10");
        // if (inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr) <= 0)
        // {
        //     ReportError("INET");
        //     break;
        // }
        // int r = connect(localFd, const &serv_addr, sizeof(serv_addr));
        int r = connect(localFd, (struct sockaddr *)&serv_addr, sizeof(serv_addr));

        if (r != 0) {
            ReportError("connect");
            *callerExitCode = ExitCode_OpenIpV4_SetSockOpt;
            break;
        }

        char *hello = "Hello from client";
        send(localFd , hello , strlen(hello) , 0); 

        retFd = localFd;
        localFd = -1;
    } while (0);

    // close(localFd);

    return retFd;
}

static void ReportError(const char *desc)
{
    Log_Debug("ERROR: TCP server: \"%s\", errno=%d (%s)\n", desc, errno, strerror(errno));
}

static ExitCode CheckNetworkStatus(void)
{
    // Ensure the necessary network interface is enabled.
    int result = Networking_SetInterfaceState(NetworkInterface, true);
    if (result != 0) {
        if (errno == EAGAIN) {
            Log_Debug("INFO: The networking stack isn't ready yet, will try again later.\n");
            return ExitCode_Success;
        } else {
            Log_Debug(
                "ERROR: Networking_SetInterfaceState for interface '%s' failed: errno=%d (%s)\n",
                NetworkInterface, errno, strerror(errno));
            return ExitCode_CheckStatus_SetInterfaceState;
        }
    }
    isNetworkStackReady = true;

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

#endif  /* NETWORK_H_ */