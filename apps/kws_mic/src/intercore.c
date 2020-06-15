#include "lib/Print.h"

#include "include/intercore.h"
#include "include/config.h"

void InterCoreInit(void) {
  IntercoreResult icr = SetupIntercoreComm(&icc, InterCoreMessageReceiveHandler);
	if (icr != Intercore_OK) {
		#if AS_DEBUG
		UART_Printf(debug, "SetupIntercoreComm failed\r\n");
		#endif
	}
}

//TODO: check for loop
void InterCoreMessageReceiveHandler(void) { 
  	for (;;) {
		ComponentId sender;
		uint8_t rxData[32];
		size_t rxDataSize = sizeof(rxData);

		IntercoreResult icr = IntercoreRecv(&icc, &sender, rxData, &rxDataSize);

		// Return if read all messages in buffer.
		if (icr == Intercore_Recv_NoBlockSize) {
				return;
		}

		// Return if an error occurred.
		if (icr != Intercore_OK) {
			#if AS_DEBUG
			UART_Print(debug, "Error in IntercoreRecv\r\n");
			return;
			#endif
		}
		#if AS_DEBUG
		UART_Printf(debug, "Message size: %d\r\n", (int)rxDataSize);
		#endif
	}
}

void InterCoreMessageSend(ComponentId* compID, void* message, uint16_t len)
{
	IntercoreResult icr;
	char* data = (char *)message;
	if (len > INTERCORE_MAX_PAYLOAD_LEN) {
		int chunk_size = 256;
		int num_chunks = len / chunk_size;
		int last_chunk = len % chunk_size;
		for(int i=0; i<num_chunks; i++) {
			while(InterCoreGetAvailableSpace() < chunk_size){ 
				;;
			}
			#if AS_DEBUG
			UART_Printf(debug, "TX loop i: %d\r\n", i);
			#endif
			icr = IntercoreSend(&icc, compID, (data + i*chunk_size), chunk_size);
			if (icr != Intercore_OK) {
				UART_Printf(debug, "IntercoreSend failed\r\n");
				return;
			}
		}
		//send rest
		while(InterCoreGetAvailableSpace() < last_chunk){ 
			;;
		}
		icr = IntercoreSend(&icc, compID, (data + num_chunks*chunk_size), last_chunk);
		if (icr != Intercore_OK) {
			UART_Printf(debug, "IntercoreSend failed\r\n");
			return;
		}
		else {
			#if AS_DEBUG
			UART_Printf(debug, "Last chunk sent!\r\n");
			#endif
		}
	}
}

int InterCoreGetAvailableSpace(void) {
	return IntercoreGetAvailableSpace(&icc);
}