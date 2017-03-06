// ....................................................................
// mp7vv.cl

#define STEPS_PER_RENDER 30
#define DELTA_T (0.002f)

float4 getforce(float4 pos, float4 vel, float dist)
{
  int eps_option[2] = {6, 3};
  float4 force;
  force.x = 0.0f;
  force.y = (-0.01)*(1-dist)*eps_option[(int)ceil(pos.y-0.5)];
  force.z = 0.0f;
  force.w = 1.0f;
  return(force);
}

float fmodulus(float n1, float n2)
{
   float divide = n1/n2;
   int trunk = (int) divide;
   float sub = divide - (float) trunk;
   float modulus = sub*n2;
   return modulus;
}

// A pseudo-random number generator - from Steve Stuart.
float goober(float ri)
{
  int hi, lo, tt;
  int a = 16807; 
  int m = 2147483647;
  int q = 127773;
  int r = 2836;
  float fm = 2147483647.0;

  int x;
  x = (int)(ri*m+0.5);
  hi = x / q;
  lo = x % q;
  tt = a * lo - r * hi;
  if (tt > 0) x = tt;
  else x = tt + m;
  return ((float)(x)/fm);
}

__kernel void hourglass(__global float4* p, __global float4* v, __global float* r, __global float* final_pos)
{
  unsigned int i = get_global_id(0);
  float4 force;
  float radius, distance;
  float offset[2] = {(1.0f-p[i].y)*0.05f, -(1.0f-p[i].y)*0.05f};

  distance = sqrt((p[i].x-0.5)*(p[i].x-0.5) + (p[i].z-0.5)*(p[i].z-0.5)); 

  for(int steps=0;steps<STEPS_PER_RENDER;steps++)
  {
    force = getforce(p[i],v[i], distance);
    v[i] += force*(DELTA_T/2.0f);
    p[i] += v[i]*DELTA_T;
    force = getforce(p[i],v[i], distance);
    v[i] += force*DELTA_T/2.0f;

    // radius = sqrt(p[i].y-0.5f)*0.375f;
    // Using this can give a more curved visual thus making it more realistic
    // for our purpose but we could not draw the external glass surface in 
    // OpenGL in the curved form to match this internal flow motion we decided
    // to make it cone shaped for consistency purposes
    distance = sqrt((p[i].x-0.5)*(p[i].x-0.5) + (p[i].z-0.5)*(p[i].z-0.5)); 

    if(p[i].y >= 0.515f)
    {
      radius = (p[i].y-0.5f)*0.375f*1.6f;
      if(distance >= radius)
      {
	// to make sure that particles converge
	p[i].x = p[i].x+0.4*offset[(int)ceil(p[i].x-0.5)];
	p[i].z = p[i].z+0.4*offset[(int)ceil(p[i].z-0.5)];
      }
    }
    else if(p[i].y <= final_pos[i])
    {
      // Stop motion.
      v[i] = (float4)(0.0f,0.0f,0.0f,1.0f);
    }
    else if((p[i].y-final_pos[i]) < 0.02f)
    { 
      // build heap
      radius = (1-(p[i].y/0.5))*(1-(p[i].y/0.5))*0.375f*0.75f;
      r[i] = goober(r[i]);
      p[i].x = goober(r[i]);
      r[i] = goober(r[i]);
      p[i].z = goober(r[i]);
      distance = sqrt((p[i].x-0.5)*(p[i].x-0.5) + (p[i].z-0.5)*(p[i].z-0.5)); 
      if(distance > radius)
      {
        p[i].x = fmodulus(p[i].x, 0.005f)+0.4975f;
        p[i].z = fmodulus(p[i].z, 0.005f)+0.4975f;
      }
    }
    else
    {
      p[i].x = fmodulus(p[i].x, 0.005f)+0.4975f;
      p[i].z = fmodulus(p[i].z, 0.005f)+0.4975f;
    }      
  }
  p[i].w = 1.0f;
}

