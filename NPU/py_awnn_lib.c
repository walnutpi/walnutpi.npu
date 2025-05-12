#include <Python.h>
#include <stdint.h>
#include <numpy/arrayobject.h>
#include <pthread.h>
#include "awnn_lib.h"
#include "awnn_internal.h"
pthread_t thread_id = 0;

int flag_thread_awnn_running = 0;
// 0: 没有运行 1:主线程已经启动了子线程 2:子线程开始运行

void *thread_run_awnn(Awnn_Context_t *context_ptr)
{
    if (flag_thread_awnn_running == 1)
    {
        flag_thread_awnn_running = 2;

        if (context_ptr != NULL)
            awnn_run(context_ptr);
        flag_thread_awnn_running = 0;
    }

    return NULL;
}

void define_commons(PyObject *module)
{
}
static PyObject *py_awnn_init(PyObject *self, PyObject *args)
{
    FILE *fp;
    char buffer[1024];
    char *model;
    fp = fopen("/proc/device-tree/model", "r");
    if (fp == NULL)
    {
        printf("Failed to open /proc/device-tree/model \n");
        exit(-1);
    }

    fgets(buffer, 100, fp);
    fclose(fp);

    model = strtok(buffer, "\n");
    if (strcmp(model, "walnutpi-2b") == 0)
        awnn_init();

    Py_RETURN_NONE;
}
static PyObject *py_awnn_uninit(PyObject *self, PyObject *args)
{
    awnn_uninit();
    Py_RETURN_NONE;
}
static PyObject *py_awnn_create(PyObject *self, PyObject *args)
{
    char *model_path;
    Awnn_Context_t *context_ptr;

    PyArg_ParseTuple(args, "s", &model_path);
    context_ptr = awnn_create(model_path);

    return Py_BuildValue("l", context_ptr);
}
static PyObject *py_awnn_destroy(PyObject *self, PyObject *args)
{
    Awnn_Context_t *context_ptr;
    if (!PyArg_ParseTuple(args, "l", &context_ptr))
        awnn_destroy(context_ptr);
    Py_RETURN_NONE;
}

static PyObject *input_buffer_pyobject = NULL;
static PyObject *py_awnn_set_input_buffers(PyObject *self, PyObject *args)
{
    PyObject *py_bytearray;
    Awnn_Context_t *context_ptr;
    if (!PyArg_ParseTuple(args, "lO!", &context_ptr, &PyByteArray_Type, &py_bytearray))
        return NULL;

    // Increase reference count to prevent Python from deallocating the bytearray
    Py_INCREF(py_bytearray);

    uint8_t *input_buffer_ptr = NULL;
    input_buffer_ptr = (uint8_t *)PyByteArray_AsString(py_bytearray);
    if (input_buffer_ptr == NULL)
    {
        // Decrease reference count if allocation fails
        Py_DECREF(py_bytearray);
        return NULL;
    }

    if (context_ptr != NULL)
    {
        awnn_set_input_buffers(context_ptr, &input_buffer_ptr);
    }

    // Store the bytearray object so it can be decremented later
    if (input_buffer_pyobject != NULL)
    {
        Py_DECREF(input_buffer_pyobject);
    }
    input_buffer_pyobject = py_bytearray;

    Py_RETURN_NONE;
}
static PyObject *py_awnn_run(PyObject *self, PyObject *args)
{
    Awnn_Context_t *context_ptr;
    PyArg_ParseTuple(args, "l", &context_ptr);
    awnn_run(context_ptr);
    Py_RETURN_NONE;
}
static PyObject *py_awnn_run_async(PyObject *self, PyObject *args)
{
    Awnn_Context_t *context_ptr;
    PyArg_ParseTuple(args, "l", &context_ptr);
    if (flag_thread_awnn_running == 0)
    {
        flag_thread_awnn_running = 1;
        int result = pthread_create(&thread_id, NULL, thread_run_awnn, context_ptr);
        if (result != 0)
        {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create thread");
            Py_RETURN_NONE;
        }
        while (flag_thread_awnn_running == 1) // 等子线程已经启动了才返回
        {
            usleep(100);
        }
    }

    Py_RETURN_NONE;
}
static PyObject *py_is_awnn_async_running(PyObject *self, PyObject *args)
{
    // 返回python bool类型的true
    if (flag_thread_awnn_running == 1)
        Py_RETURN_TRUE;
    else
        Py_RETURN_FALSE;
}
static PyObject *py_awnn_get_output_buffer(PyObject *self, PyObject *args)
{
    int index;
    Awnn_Context_t *context_ptr;
    PyArg_ParseTuple(args, "li", &context_ptr, &index);
    float *output_buffer = awnn_get_output_buffers(context_ptr)[index];
    // 构建一个nunpy的数组类型进行返回
    int output_buffer_size = context_ptr->output_params[index].elements;
    npy_intp dims[1] = {output_buffer_size};
    PyObject *numpy_array = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, output_buffer);

    // Py_INCREF(output_buffer);
    return numpy_array;
}

static PyObject *py_awnn_dump_io(PyObject *self, PyObject *args)
{
    char *out_path;
    Awnn_Context_t *context_ptr;
    PyArg_ParseTuple(args, "ls", &context_ptr, &out_path);
    if (context_ptr != NULL)
        awnn_dump_io(context_ptr, out_path);
    Py_RETURN_NONE;
}
static PyObject *get_output_buffer_count(PyObject *self, PyObject *args)
{
    Awnn_Context_t *context_ptr;
    return Py_BuildValue("li", &context_ptr, context_ptr->output_count);
}

static const char moduledocstring[] = "awnn";

PyMethodDef pinctrl_methods[] = {
    {"awnn_init", py_awnn_init, METH_VARARGS, "init awnn"},
    {"awnn_uninit", py_awnn_uninit, METH_VARARGS, "uninit awnn"},
    {"awnn_create", py_awnn_create, METH_VARARGS, "create network"},
    {"awnn_destroy", py_awnn_destroy, METH_VARARGS, "destroy network"},
    {"awnn_set_input_buffers", py_awnn_set_input_buffers, METH_VARARGS, "set input buffers for network"},
    {"awnn_run", py_awnn_run, METH_VARARGS, "run network"},
    {"awnn_run_async", py_awnn_run_async, METH_VARARGS, "run network as async"},
    {"is_awnn_async_running", py_is_awnn_async_running, METH_VARARGS, "is async running"},
    {"awnn_dump_io", py_awnn_dump_io, METH_VARARGS, "dump the input and output tensors"},
    {"get_output_buffer_count", get_output_buffer_count, METH_VARARGS, "get output buffer count"},
    {"awnn_get_output_buffer", py_awnn_get_output_buffer, METH_VARARGS, "get output buffer by index"},
    {NULL, NULL, 0, NULL},
};
static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "_awnn_lib",     // name of module
    moduledocstring, // module documentation, may be NULL
    -1,              // size of per-interpreter state of the module, or -1 if the module keeps state in global variables.
    pinctrl_methods};

PyMODINIT_FUNC PyInit__awnn_lib(void)
{
    import_array();
    PyObject *module = NULL;

    if ((module = PyModule_Create(&module_def)) == NULL)
        return NULL;
    define_commons(module);

    return module;
}
