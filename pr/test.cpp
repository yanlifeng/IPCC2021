#include <cstdio>
#include <cmath>
#include <iostream>

using namespace std;
#define EXP_A 184
#define EXP_C 16249

float EXP(float y) {
    union {
        float d;
        struct {
#ifdef LITTLE_ENDIAN
            short j, i;
#else
            short i, j;
#endif
        } n;
    } eco;
    eco.n.i = EXP_A * (y) + (EXP_C);
    eco.n.j = 0;
    return eco.d;
}

float LOG(float y) {
    int *nTemp = (int *) &y;
    y = (*nTemp) >> 16;
    return (y - EXP_C) / EXP_A;
}

float POW(float b, float p) {
    return EXP(LOG(b) * p);
}

int main() {

    for (int i = 0; i < 1000; i++) {
        float r1 = sqrt(pow(i * 1.14, 24), 10);
        float r2 = pow(i * 1.14, 2.4);
        if (fabs(r1 - r2) > 1e-8)
            cout << r1 << " " << r2 << endl;
    }

}