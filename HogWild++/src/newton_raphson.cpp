#include "../include/newton_raphson.hpp"

#define EPSILON 0.001


float f(float beta, float m)
{
    return pow(beta, m) + beta - 1;
}

float df(float beta, float m)
{
    return m * pow(beta, m-1) + 1;
}

// Not really sure how this works
float newtonRaphson(float x, float m)
{
    float h = f(x, m) / df(x, m);
    while(fabs(h) >= EPSILON)
    {
        h = f(x, m) / df(x, m);
        x = x - h;
    }
    return x;
}