// 
// Hour glass simulation.
//
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <GL/glew.h>
#include <GL/glut.h>
#include <GL/glx.h>
#include <CL/cl.h>
#include <CL/cl_gl.h>
#include <CL/cl_ext.h>
#include <CL/cl_gl_ext.h>
#include <CL/cl_platform.h>
#include <CL/opencl.h>
#include "RGU.h"

GLuint OGL_VBO = 1;

#define NUMBER_OF_PARTICLES 1024*1024
#define DATA_SIZE (NUMBER_OF_PARTICLES*4*sizeof(float)) 

cl_context mycontext;
cl_command_queue mycommandqueue;
cl_kernel mykernel;
cl_program myprogram;
cl_mem oclvbo, dev_velocity, dev_rseed, dev_final_pos;
size_t worksize[] = {NUMBER_OF_PARTICLES}; 
size_t lws[] = {128}; 

float host_position[NUMBER_OF_PARTICLES][4];
float host_velocity[NUMBER_OF_PARTICLES][4];
float host_rseed[NUMBER_OF_PARTICLES];
float host_final_pos[NUMBER_OF_PARTICLES];

void do_kernel()
{
cl_event waitlist[1];
clEnqueueNDRangeKernel(mycommandqueue,mykernel,1,NULL,worksize,lws,0,0,
	&waitlist[0]);
clWaitForEvents(1,waitlist);
}

void do_material_points()
{
float mat_ambient[] = {0.0,0.0,0.0,1.0};
float mat_diffuse[] = {1.0,1.0,0.1,1.0};
float mat_specular[] = {1.0,1.0,1.0,1.0};
float mat_shininess[] = {2.0};

glMaterialfv(GL_FRONT,GL_AMBIENT,mat_ambient);
glMaterialfv(GL_FRONT,GL_DIFFUSE,mat_diffuse);
glMaterialfv(GL_FRONT,GL_SPECULAR,mat_specular);
glMaterialfv(GL_FRONT,GL_SHININESS,mat_shininess);
}

void update()
{
glFinish();
clEnqueueAcquireGLObjects(mycommandqueue,1,&oclvbo,0,0,0);
do_kernel();
clEnqueueReleaseGLObjects(mycommandqueue, 1, &oclvbo, 0,0,0);
clFinish(mycommandqueue);
glutPostRedisplay();
}

void mydisplayfunc()
{
glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

glColorMaterial ( GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE );
glEnable ( GL_COLOR_MATERIAL );
glPushMatrix();
glTranslatef(0.0,-0.25,0.0);
glPushMatrix();
glTranslatef(0.0,0.425,0.0);
glScalef(0.72,0.68,0.7);

 GLfloat xRotated=80.0;
 GLfloat yRotated=10.0;
 GLfloat zRotated=140.0;

//Left and right cylinders
glPushMatrix();
GLUquadricObj *quadratic;
quadratic = gluNewQuadric();
glRotatef(xRotated,1.0,0.0,0.0);
glRotatef(yRotated,0.0,1.0,0.0);
glRotatef(zRotated,0.0,0.0,1.0);
glTranslatef(0.78,0.0,-1.05);
glColor3f(0.87,0.69,0.22);
gluCylinder(quadratic,0.01f,0.01f,2.55f,50,50);
glPopMatrix();

glPushMatrix();
glRotatef(xRotated,1.0,0.0,0.0);
glRotatef(yRotated,0.0,1.0,0.0);
glRotatef(zRotated,0.0,0.0,1.0);
glTranslatef(-0.75,0.0,-1.04);
glColor3f(0.87,0.69,0.22);
gluCylinder(quadratic,0.01f,0.01f,2.5f,50,50);
glPopMatrix();

//Top and bottom cylinders
glPushMatrix();
glRotatef(xRotated,1.0,0.0,0.0);
glRotatef(yRotated,0.0,1.0,0.0);
glRotatef(zRotated,0.0,0.0,1.0);
glTranslatef(0.0,0.0,1.2);
glColor3f(0.87,0.69,0.22);
gluCylinder(quadratic,0.75,0.75,0.2,50,50);
glRotatef(180, 1,0,0);
gluDisk(quadratic, 0.0f, 0.75, 50, 1);
glRotatef(180, 1,0,0);
glTranslatef(0.0f, 0.0f, 0.2);
gluDisk(quadratic, 0.0f, 0.75, 50, 1);
glTranslatef(0.0f, 0.0f, -0.2);
glPopMatrix();

glPushMatrix();
glRotatef(xRotated,1.0,0.0,0.0);
glRotatef(yRotated,0.0,1.0,0.0);
glRotatef(zRotated,0.0,0.0,1.0);
glTranslatef(0.0,0.0,-1.0);
glColor3f(0.87,0.69,0.22);
gluCylinder(quadratic,0.75,0.75,0.2,50,50);
glRotatef(180, 1,0,0);
gluDisk(quadratic, 0.0f, 0.75, 50, 1);
glRotatef(180, 1,0,0);
glTranslatef(0.0f, 0.0f, 0.2);
gluDisk(quadratic, 0.0f, 0.75, 50, 1);
glTranslatef(0.0f, 0.0f, -0.2);
glPopMatrix();

//Transparent material
glEnable(GL_BLEND);
glColor4f(1.0f, 1.0f, 1.0f, 0.5);
glBlendFunc(GL_SRC_ALPHA, GL_ONE);

GLdouble top=0.75;
GLdouble bottom=0.1;
GLdouble height=1.0;
glPushMatrix();
quadratic = gluNewQuadric();
glRotatef(xRotated,1.0,0.0,0.0);
glRotatef(yRotated,0.0,1.0,0.0);
glRotatef(zRotated,0.0,0.0,1.0);
glTranslatef(0.0,0.0,-0.8);
glColor3f(0.5,0.5,0.5);
gluCylinder(quadratic,top,bottom,height,50,50);
glPopMatrix();

glPushMatrix();
quadratic = gluNewQuadric();
glRotatef(xRotated,1.0,0.0,0.0);
glRotatef(yRotated,0.0,1.0,0.0);
glRotatef(zRotated,0.0,0.0,1.0);
glTranslatef(0.0,0.0,0.2);
glColor3f(0.5,0.5,0.5);
gluCylinder(quadratic,bottom,top,height,50,50);
glPopMatrix();

glDisable(GL_BLEND);

glPopMatrix();
glDisable(GL_COLOR_MATERIAL);
glEnable(GL_LIGHTING);
do_material_points();
glBindBuffer(GL_ARRAY_BUFFER,OGL_VBO);
glVertexPointer(4,GL_FLOAT,0,0);
glEnableClientState(GL_VERTEX_ARRAY);
glDrawArrays(GL_POINTS, 0, NUMBER_OF_PARTICLES);
glDisableClientState(GL_VERTEX_ARRAY);
glutSwapBuffers();
glPopMatrix();
}

void setup_the_viewvol()
{
float eye[] = {2.0, 1.0, 2.0};
float view[] = {0.0, 0.0, 0.0};
float up[] = {0.0, 1.0, 0.0};

glMatrixMode(GL_PROJECTION);
glLoadIdentity();
gluPerspective(45.0,1.0,0.1,20.0);

glMatrixMode(GL_MODELVIEW);
glLoadIdentity();
gluLookAt(eye[0],eye[1],eye[2],view[0],view[1],view[2],up[0],up[1],up[2]);
}

void do_lights()
{
float light_ambient[] = { 0.0, 0.0, 0.0, 0.0 };
float light_diffuse[] = { 0.8, 0.8, 0.8, 0.0 };
float light_specular[] = { 0.1, 0.1, 0.1, 1.0 };
float light_position[] = { 2.0, 2.0, 2.0, 1.0 };
float light_direction[] = { -1.0, -1.0, -1.0, 1.0};

glLightModelfv(GL_LIGHT_MODEL_AMBIENT,light_ambient);
glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER,1);

glLightfv(GL_LIGHT0,GL_AMBIENT,light_ambient);
glLightfv(GL_LIGHT0,GL_DIFFUSE,light_diffuse);
glLightfv(GL_LIGHT0,GL_SPECULAR,light_specular);
glLightf(GL_LIGHT0,GL_SPOT_EXPONENT,1.0);
glLightf(GL_LIGHT0,GL_SPOT_CUTOFF,180.0);
glLightf(GL_LIGHT0,GL_CONSTANT_ATTENUATION,0.5);
glLightf(GL_LIGHT0,GL_LINEAR_ATTENUATION,0.1);
glLightf(GL_LIGHT0,GL_QUADRATIC_ATTENUATION,0.01);
glLightfv(GL_LIGHT0,GL_POSITION,light_position);
glLightfv(GL_LIGHT0,GL_SPOT_DIRECTION,light_direction);

glEnable(GL_LIGHTING);
glEnable(GL_LIGHT0);
}

void InitGL(int argc, char** argv)
{
glutInit(&argc,argv);
glutInitDisplayMode(GLUT_RGBA|GLUT_DEPTH|GLUT_DOUBLE|GLUT_MULTISAMPLE);
glutInitWindowSize(768,768);
glutInitWindowPosition(100,50);
glutCreateWindow("my_cool_cube");
setup_the_viewvol();
do_lights();
glPointSize(1.0);
glLineWidth(3.0);
glEnable(GL_DEPTH_TEST);
glEnable(GL_MULTISAMPLE_ARB);
glClearColor(0.1,0.2,0.35,1.0);
glewInit();
return;
}

double genrand()
{
//generates random number between 0.0 and 1.0
return (((double)(random()+1))/2147483649.);
}

void init_particles()
{
int i, j;
for(i=0;i<NUMBER_OF_PARTICLES;i++){
        double x, z;
	double ans = 1.0;
	while(ans > 0.3)
	{
		x = genrand();
		z = genrand();
		ans = sqrt((x-0.5)*(x-0.5)+(z-0.5)*(z-0.5));
	}
	host_position[i][0] = x;
	host_position[i][1] = 1.0f-0.5*genrand();
	host_final_pos[i] = (host_position[i][1]-0.5f)*0.58;
	host_position[i][2] = z;
	host_position[i][3] = 1.0;
	for(j=0;j<4;j++) host_velocity[i][j] = 0.0;
	host_rseed[i] = genrand();
	}
}

void InitCL()
{
cl_platform_id myplatform;
cl_device_id *mydevice;
cl_int err;
char* oclsource; 
size_t program_length;
unsigned int gpudevcount;

err = RGUGetPlatformID(&myplatform);

err = clGetDeviceIDs(myplatform,CL_DEVICE_TYPE_GPU,0,NULL,&gpudevcount);
mydevice = new cl_device_id[gpudevcount];
err = clGetDeviceIDs(myplatform,CL_DEVICE_TYPE_GPU,gpudevcount,mydevice,NULL);

// You need all these to get full interoperability with OpenGL:
cl_context_properties props[] = {
        CL_GL_CONTEXT_KHR, (cl_context_properties)glXGetCurrentContext(),
        CL_GLX_DISPLAY_KHR, (cl_context_properties)glXGetCurrentDisplay(),
        CL_CONTEXT_PLATFORM, (cl_context_properties)myplatform,
        0};

mycontext = clCreateContext(props,1,&mydevice[0],NULL,NULL,&err);
mycommandqueue = clCreateCommandQueue(mycontext,mydevice[0],0,&err);

oclsource = RGULoadProgSource("hourglass.cl", "", &program_length);
myprogram = clCreateProgramWithSource(mycontext,1,(const char **)&oclsource,
	&program_length, &err);
if(err==CL_SUCCESS) fprintf(stderr,"create ok\n");
else fprintf(stderr,"create err %d\n",err);
clBuildProgram(myprogram, 0, NULL, NULL, NULL, NULL);
mykernel = clCreateKernel(myprogram, "hourglass", &err);
if(err==CL_SUCCESS) fprintf(stderr,"build ok\n");
else fprintf(stderr,"build err %d\n",err);

glBindBuffer(GL_ARRAY_BUFFER, OGL_VBO);
glBufferData(GL_ARRAY_BUFFER, DATA_SIZE, &host_position[0][0], GL_DYNAMIC_DRAW);
oclvbo = clCreateFromGLBuffer(mycontext,CL_MEM_WRITE_ONLY,OGL_VBO,&err);

dev_velocity = clCreateBuffer(mycontext,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,
	DATA_SIZE,&host_velocity[0][0],&err); 

dev_rseed = clCreateBuffer(mycontext,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,
	NUMBER_OF_PARTICLES*sizeof(float),&host_rseed[0],&err); 

dev_final_pos = clCreateBuffer(mycontext,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
	NUMBER_OF_PARTICLES*sizeof(float),&host_final_pos[0],&err); 

clSetKernelArg(mykernel,0,sizeof(cl_mem),(void *)&oclvbo);
clSetKernelArg(mykernel,1,sizeof(cl_mem),(void *)&dev_velocity);
clSetKernelArg(mykernel,2,sizeof(cl_mem),(void *)&dev_rseed);
clSetKernelArg(mykernel,3,sizeof(cl_mem),(void *)&dev_final_pos);
}

void cleanup()
{
clReleaseKernel(mykernel);
clReleaseProgram(myprogram);
clReleaseCommandQueue(mycommandqueue);
glBindBuffer(GL_ARRAY_BUFFER,OGL_VBO);
glDeleteBuffers(1,&OGL_VBO);
clReleaseMemObject(oclvbo);
clReleaseMemObject(dev_velocity);
clReleaseMemObject(dev_rseed);
clReleaseContext(mycontext);
exit(0);
}

void getout(unsigned char key, int x, int y)
{
switch(key) {
        case 'q':
                cleanup();
        default:
                break;
    }
}

int main(int argc,char **argv)
{
srandom(123456789);

//change init particles
init_particles();
InitGL(argc, argv); 
InitCL(); 
glutDisplayFunc(mydisplayfunc);
glutIdleFunc(update);
glutKeyboardFunc(getout);
glutMainLoop();
}

