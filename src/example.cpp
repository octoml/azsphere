
#include "example.h"

EXAMPLE::EXAMPLE() {
    _test = 10;
}

void EXAMPLE::Print() {
    fprintf(stdout, "example variable: %d\n", _test);
}
