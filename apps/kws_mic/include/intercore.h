#ifndef AS_INTERCORE_H_
#define AS_INTERCORE_H_

#include "lib/logical-intercore.h"

IntercoreComm icc;

void InterCoreInit(void);
void InterCoreMessageReceiveHandler(void);
void InterCoreMessageSend(ComponentId* compID, void* message, uint16_t len);
int InterCoreGetAvailableSpace(void);

#endif /* AS_INTERCORE_H_ */