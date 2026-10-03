/* Historical scalar-libm and eight-lane CPU reduction order.
 * Formulas/reference: PyTorch v2.0.1 moments_utils.h and SoftMaxKernel.cpp.
 * PyTorch source license/attribution is retained in validation/PYTORCH_LICENSE.txt.
 * This implementation does not link or load historical PyTorch binaries. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <string.h>

static int buffer(PyObject *object, Py_buffer *view, int writable) {
    if (PyObject_GetBuffer(object, view, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT | (writable ? PyBUF_WRITABLE : 0)) < 0) return 0;
    if (view->itemsize != sizeof(float) || !view->format || strcmp(view->format,"f")) {
        PyErr_SetString(PyExc_ValueError,"Expected contiguous native float32 buffer");
        PyBuffer_Release(view); view->obj=NULL; return 0;
    }
    return 1;
}
static void merge(int add, const float *am, const float *av, int *count, float *mean, float *var, int lanes) {
    int total=*count+add; float c=total ? (float)add/(float)total : 0.f;
    for (int j=0;j<lanes;j++) {float d=am[j]-mean[j];mean[j]+=c*d;var[j]+=av[j]+((d*d)*c)*(float)*count;}
    *count=total;
}
static PyObject *moments(PyObject *self, PyObject *args) {
    PyObject *source,*means,*inverse; Py_ssize_t columns; double eps;
    Py_buffer x={0},m={0},v={0};
    if (!PyArg_ParseTuple(args,"OOOnd",&source,&means,&inverse,&columns,&eps)) return NULL;
    if (!buffer(source,&x,0) || !buffer(means,&m,1) || !buffer(inverse,&v,1)) goto fail;
    if (columns<=0 || columns>1048576 || x.len%(sizeof(float)*columns) || m.len!=v.len || m.len/sizeof(float)!=x.len/(sizeof(float)*columns)) {
        PyErr_SetString(PyExc_ValueError,"Invalid rowwise-moments buffer sizes");goto fail;
    }
    Py_ssize_t rows=m.len/sizeof(float);const float *input=x.buf;float *output_mean=m.buf,*output_inv=v.buf;
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t row=0;row<rows;row++) {
        const float *values=input+row*columns;int nv=(int)(columns/8),chunks=(nv+15)/16,depth=1;
        while ((1<<depth)<chunks) depth++;
        int counts[32]={0};float ms[32][8]={{0}},vs[32][8]={{0}};
        for (int i=0;i<chunks;i++) {
            int count=nv-i*16;if(count>16)count=16;
            float mean[8]={0},var[8]={0};
            for (int j=0;j<count;j++) for(int k=0;k<8;k++) {
                float value=values[(i*16+j)*8+k],delta=value-mean[k];
                mean[k]+=delta*(1.f/(float)(j+1));var[k]+=delta*(value-mean[k]);
            }
            merge(count,mean,var,&counts[0],ms[0],vs[0],8);
            int mask=i+1;
            for(int level=1;level<depth && !(mask&1);level++,mask>>=1) {
                merge(counts[level-1],ms[level-1],vs[level-1],&counts[level],ms[level],vs[level],8);
                counts[level-1]=0;memset(ms[level-1],0,sizeof(ms[0]));memset(vs[level-1],0,sizeof(vs[0]));
            }
        }
        for(int level=1;level<depth;level++)merge(counts[level],ms[level],vs[level],&counts[0],ms[0],vs[0],8);
        int count=0;float mean=0.f,var=0.f;
        for(Py_ssize_t i=nv*8;i<columns;i++) {float d=values[i]-mean;count++;mean+=d/(float)count;var+=d*(values[i]-mean);}
        for(int k=0;k<8;k++)merge(nv,&ms[0][k],&vs[0][k],&count,&mean,&var,1);
        output_mean[row]=mean;output_inv[row]=1.f/sqrtf(var/(float)columns+(float)eps);
    }
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&x);PyBuffer_Release(&m);PyBuffer_Release(&v);Py_RETURN_NONE;
fail:
    if(x.obj)PyBuffer_Release(&x);if(m.obj)PyBuffer_Release(&m);if(v.obj)PyBuffer_Release(&v);return NULL;
}
static float lane_sum(const float *x, Py_ssize_t width, int exponential, float maximum) {
    float lanes[8]={0};
    for(Py_ssize_t i=0;i<width;i++) lanes[i%8]+=exponential ? expf(x[i]-maximum) : x[i];
    float result=lanes[0];for(int k=1;k<8 && k<width;k++)result+=lanes[k];return result;
}
static PyObject *logsoftmax(PyObject *self, PyObject *args) {
    PyObject *source,*dest;Py_ssize_t width;Py_buffer x={0},y={0};
    if(!PyArg_ParseTuple(args,"OOn",&source,&dest,&width))return NULL;
    if(!buffer(source,&x,0)||!buffer(dest,&y,1))goto fail;
    if(width<=0||width>x.len/sizeof(float)||x.len!=y.len||x.len%(width*sizeof(float))) {PyErr_SetString(PyExc_ValueError,"Invalid log-softmax buffer sizes");goto fail;}
    const float *input=x.buf;float *output=y.buf;Py_ssize_t rows=x.len/sizeof(float)/width;
    Py_BEGIN_ALLOW_THREADS
    for(Py_ssize_t r=0;r<rows;r++) {
        const float *row=input+r*width;float maximum=row[0];for(Py_ssize_t i=1;i<width;i++)if(row[i]>maximum)maximum=row[i];
        float log_sum=logf(lane_sum(row,width,1,maximum));
        for(Py_ssize_t i=0;i<width;i++)output[r*width+i]=(row[i]-maximum)-log_sum;
    }
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&x);PyBuffer_Release(&y);Py_RETURN_NONE;
fail:if(x.obj)PyBuffer_Release(&x);if(y.obj)PyBuffer_Release(&y);return NULL;
}
static PyObject *logsoftmax_backward(PyObject *self, PyObject *args) {
    PyObject *gradient,*result,*dest;Py_ssize_t width;Py_buffer g={0},o={0},d={0};
    if(!PyArg_ParseTuple(args,"OOOn",&gradient,&result,&dest,&width))return NULL;
    if(!buffer(gradient,&g,0)||!buffer(result,&o,0)||!buffer(dest,&d,1))goto fail;
    if(width<=0||width>g.len/sizeof(float)||g.len!=o.len||g.len!=d.len||g.len%(width*sizeof(float))){PyErr_SetString(PyExc_ValueError,"Invalid backward buffer sizes");goto fail;}
    const float *grad=g.buf,*output=o.buf;float *dx=d.buf;Py_ssize_t rows=g.len/sizeof(float)/width;
    Py_BEGIN_ALLOW_THREADS
    for(Py_ssize_t r=0;r<rows;r++) {
        float sum=lane_sum(grad+r*width,width,0,0.f);
        for(Py_ssize_t i=0;i<width;i++) {Py_ssize_t at=r*width+i;dx[at]=grad[at]-expf(output[at])*sum;}
    }
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&g);PyBuffer_Release(&o);PyBuffer_Release(&d);Py_RETURN_NONE;
fail:if(g.obj)PyBuffer_Release(&g);if(o.obj)PyBuffer_Release(&o);if(d.obj)PyBuffer_Release(&d);return NULL;
}

static PyObject *softmax(PyObject *self, PyObject *args) {
    PyObject *source,*dest;Py_ssize_t width;Py_buffer x={0},y={0};
    if(!PyArg_ParseTuple(args,"OOn",&source,&dest,&width))return NULL;
    if(!buffer(source,&x,0)||!buffer(dest,&y,1))goto fail;
    if(width<=0||width>x.len/sizeof(float)||x.len!=y.len||x.len%(width*sizeof(float))){PyErr_SetString(PyExc_ValueError,"Invalid softmax buffer sizes");goto fail;}
    const float *input=x.buf;float *output=y.buf;Py_ssize_t rows=x.len/sizeof(float)/width;
    Py_BEGIN_ALLOW_THREADS
    for(Py_ssize_t r=0;r<rows;r++) {
        const float *row=input+r*width;float maximum=row[0];for(Py_ssize_t i=1;i<width;i++)if(row[i]>maximum)maximum=row[i];
        float sum=lane_sum(row,width,1,maximum);float inverse=1.f/sum;
        for(Py_ssize_t i=0;i<width;i++)output[r*width+i]=expf(row[i]-maximum)*inverse;
    }
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&x);PyBuffer_Release(&y);Py_RETURN_NONE;
fail:if(x.obj)PyBuffer_Release(&x);if(y.obj)PyBuffer_Release(&y);return NULL;
}
static PyObject *softmax_backward(PyObject *self, PyObject *args) {
    PyObject *gradient,*result,*dest;Py_ssize_t width;Py_buffer g={0},o={0},d={0};
    if(!PyArg_ParseTuple(args,"OOOn",&gradient,&result,&dest,&width))return NULL;
    if(!buffer(gradient,&g,0)||!buffer(result,&o,0)||!buffer(dest,&d,1))goto fail;
    if(width<=0||width>g.len/sizeof(float)||g.len!=o.len||g.len!=d.len||g.len%(width*sizeof(float))){PyErr_SetString(PyExc_ValueError,"Invalid softmax backward buffer sizes");goto fail;}
    const float *grad=g.buf,*output=o.buf;float *dx=d.buf;Py_ssize_t rows=g.len/sizeof(float)/width;
    Py_BEGIN_ALLOW_THREADS
    for(Py_ssize_t r=0;r<rows;r++) {
        float lanes[8]={0};
        for(Py_ssize_t i=0;i<width;i++) {Py_ssize_t at=r*width+i;lanes[i%8]+=grad[at]*output[at];}
        float sum=lanes[0];for(int k=1;k<8 && k<width;k++)sum+=lanes[k];
        for(Py_ssize_t i=0;i<width;i++) {Py_ssize_t at=r*width+i;dx[at]=(grad[at]-sum)*output[at];}
    }
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&g);PyBuffer_Release(&o);PyBuffer_Release(&d);Py_RETURN_NONE;
fail:if(g.obj)PyBuffer_Release(&g);if(o.obj)PyBuffer_Release(&o);if(d.obj)PyBuffer_Release(&d);return NULL;
}
static PyMethodDef methods[]={{"softmax",softmax,METH_VARARGS,"Historical scalar-libm softmax."},{"softmax_backward",softmax_backward,METH_VARARGS,"Historical first-order softmax derivative."},{"moments",moments,METH_VARARGS,"Historical eight-lane row moments."},{"logsoftmax",logsoftmax,METH_VARARGS,"Historical scalar-libm log-softmax."},{"logsoftmax_backward",logsoftmax_backward,METH_VARARGS,"Historical first-order log-softmax derivative."},{NULL,NULL,0,NULL}};
static struct PyModuleDef module={PyModuleDef_HEAD_INIT,"_legacy_math",NULL,-1,methods};
PyMODINIT_FUNC PyInit__legacy_math(void){return PyModule_Create(&module);}
