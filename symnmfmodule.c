
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "symnmf.h"

// Convert Python list to C double array
static double** py_to_c_matrix(PyObject* py_matrix, int* rows, int* cols) {
    PyObject* row;
    *rows = PyList_Size(py_matrix);
    *cols = PyList_Size(PyList_GetItem(py_matrix, 0));
    double** c_matrix = (double**)malloc(*rows * sizeof(double*));
    if (!c_matrix) {
        return NULL;
    }
    for (int i = 0; i < *rows; i++) {
        row = PyList_GetItem(py_matrix, i);
        c_matrix[i] = (double*)malloc(*cols * sizeof(double));
        if (!c_matrix[i]) {
            return NULL;
        }
        for (int j = 0; j < *cols; j++) {
            c_matrix[i][j] = PyFloat_AsDouble(PyList_GetItem(row, j));
        }
    }
    return c_matrix;
}

// Convert C double array to Python list
static PyObject* c_to_py_matrix(double** c_matrix, int rows, int cols) {
    PyObject* py_matrix = PyList_New(rows);
    for (int i = 0; i < rows; i++) {
        PyObject* py_row = PyList_New(cols);
        for (int j = 0; j < cols; j++) {
            PyList_SetItem(py_row, j, PyFloat_FromDouble(c_matrix[i][j]));
        }
        PyList_SetItem(py_matrix, i, py_row);
    }
    return py_matrix;
}

// sym function: calculates the similarity matrix
static PyObject* py_sym(PyObject* self, PyObject* args) {
    PyObject* py_matrix;
    if (!PyArg_ParseTuple(args, "O", &py_matrix)) {
        return NULL;
    }
    int rows, cols;
    double** c_matrix = py_to_c_matrix(py_matrix, &rows, &cols);
    if (!c_matrix) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for matrix.");
        return NULL;
    }
    double** result = sym(c_matrix, rows, cols);
    PyObject* py_result = c_to_py_matrix(result, rows, rows); // similarity matrix is square
    free_matrix(c_matrix, rows);
    free_matrix(result, rows);
    return py_result;
}

// ddg function: calculates the diagonal degree matrix
static PyObject* py_ddg(PyObject* self, PyObject* args) {
    PyObject* py_matrix;
    if (!PyArg_ParseTuple(args, "O", &py_matrix)) {
        return NULL;
    }
    int rows, cols;
    double** c_matrix = py_to_c_matrix(py_matrix, &rows, &cols);
    if (!c_matrix) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for matrix.");
        return NULL;
    }
    double** result = ddg(c_matrix, rows);
    PyObject* py_result = c_to_py_matrix(result, rows, rows);
    free_matrix(c_matrix, rows);
    free_matrix(result, rows);
    return py_result;
}

// norm function: calculates the normalized similarity matrix
static PyObject* py_norm(PyObject* self, PyObject* args) {
    PyObject* py_matrix;
    if (!PyArg_ParseTuple(args, "O", &py_matrix)) {
        return NULL;
    }
    int rows, cols;
    double** c_matrix = py_to_c_matrix(py_matrix, &rows, &cols);
    if (!c_matrix) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for matrix.");
        return NULL;
    }
    double** result = norm(c_matrix, rows);
    PyObject* py_result = c_to_py_matrix(result, rows, rows);
    free_matrix(c_matrix, rows);
    free_matrix(result, rows);
    return py_result;
}

// symnmf function: performs the SymNMF algorithm
static PyObject* py_symnmf(PyObject* self, PyObject* args) {
    PyObject* py_matrix;
    PyObject* py_H;
    int k;
    if (!PyArg_ParseTuple(args, "OOi", &py_matrix, &py_H, &k)) {
        return NULL;
    }
    int rows, cols, h_rows, h_cols;
    double** c_matrix = py_to_c_matrix(py_matrix, &rows, &cols);
    double** c_H = py_to_c_matrix(py_H, &h_rows, &h_cols);
    if (!c_matrix || !c_H) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for matrices.");
        return NULL;
    }
    double** result = symnmf(c_matrix, c_H, rows, cols, k);
    PyObject* py_result = c_to_py_matrix(result, rows, k);
    free_matrix(c_matrix, rows);
    free_matrix(c_H, h_rows);
    free_matrix(result, rows);
    return py_result;
}

// Method definitions
static PyMethodDef SymNMFMethods[] = {
    {"sym", py_sym, METH_VARARGS, "Calculate the similarity matrix"},
    {"ddg", py_ddg, METH_VARARGS, "Calculate the diagonal degree matrix"},
    {"norm", py_norm, METH_VARARGS, "Calculate the normalized similarity matrix"},
    {"symnmf", py_symnmf, METH_VARARGS, "Perform the SymNMF algorithm"},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef symnmfmodule = {
    PyModuleDef_HEAD_INIT,
    "symnmfmodule",
    "Symmetric NMF Module",
    -1,
    SymNMFMethods
};

// Module initialization
PyMODINIT_FUNC PyInit_symnmfmodule(void) {
    return PyModule_Create(&symnmfmodule);
}
