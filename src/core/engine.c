#include <Python.h>
#include <arrayobject.h>
#include <time.h>
#include <math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_sf_erf.h>

static PyObject* g_engineError;
static gsl_rng* g_rng;

/** low level helpers **/

/** numpy helpers **/

/**
 *  Check that PyArrayObject is a double (Float) type and a vector
 * 	return 1 if an error and raise exception.
 */
static int not_doublevector(PyArrayObject *vec)  {
	if (vec->descr->type_num != NPY_DOUBLE || vec->nd != 1)  {
		PyErr_SetString(PyExc_ValueError,
			"In not_doublevector: array must be of type Float and 1 dimensional (n).");
		return 1;
	}
	return 0;
}

static double lognormal(double m, double stddev) {
/*
compute pdf with parameters m and stddev
gsl_ran_lognormal(g_rng, mulnx, siglnx)
m: mean of x
stddev: std of x
*/
	return gsl_ran_lognormal(g_rng,
			log(m) - 0.5 * log(1.0 + (stddev*stddev)/(m*m)),
			sqrt(log((stddev*stddev)/(m*m) + 1)) );
}

/** python callables **/

static PyObject* engine_bivariate_gaussian(PyObject *self, PyObject *args) {
	float sx, sy;
	if (!PyArg_ParseTuple(args, "ff", &sx, &sy))
        return NULL;
	double x, y;
	gsl_ran_bivariate_gaussian(g_rng, sx, sy, 0.0, &x, &y);
	return Py_BuildValue("(ff)", (float)x, (float)y);
}

static PyObject* engine_beta(PyObject *self, PyObject *args) {
	float a = 0;
	float b = 0;
	if (!PyArg_ParseTuple(args, "ff", &a, &b))
        return NULL;
	double sample = (float)gsl_ran_beta(g_rng, a, b);
	return Py_BuildValue("f", (float)sample);
}

static PyObject* engine_lognormal(PyObject *self, PyObject *args) {
	float m = 0;
	float stddev = 0;
	if (!PyArg_ParseTuple(args, "ff", &m, &stddev))
        return NULL;
	return Py_BuildValue("f", (float)lognormal(m, stddev));
}	

static PyObject* engine_lognormal_cdf(PyObject *self, PyObject *args) {
	float m = 0;
	float stddev = 0;
	float x = 0;
	if (!PyArg_ParseTuple(args, "fff", &x, &m, &stddev))
        return NULL;
	double m2 = log(m) - 0.5 * log(1.0 + (stddev*stddev)/(m*m));
	double s2 = sqrt(log((stddev*stddev)/(m*m) + 1));
	return Py_BuildValue("f", (float)gsl_cdf_lognormal_P(x, m2, s2));
}

static PyObject* engine_lognormal_cdf_inv(PyObject *self, PyObject *args) {
	float m = 0;
	float stddev = 0;
	float p = 0;
	if (!PyArg_ParseTuple(args, "fff", &p, &m, &stddev))
		return NULL;
	double m2 = log(m) - 0.5 * log(1.0 + (stddev*stddev)/(m*m));
	double s2 = sqrt(log((stddev*stddev)/(m*m) + 1));
	return Py_BuildValue("f", (float)gsl_cdf_lognormal_Pinv(p, m2, s2));
}

static PyObject* engine_percentileofscore(PyObject *self, PyObject *args) {
	// python: engine.percentileofscore([scores], score)
	double score = 0;
	PyObject* listObj;
	if (!PyArg_ParseTuple(args, "O!d", &PyList_Type, &listObj, &score))
        return NULL;
	
	// loop through python list of doubles, counting scores less than or equal to score
	int arrsz = PyList_Size(listObj);
	if (arrsz == 0)
		return NULL;
	int i;
	int count=0;
	for (i=0; i<arrsz; i++) {
		double aScore = PyFloat_AsDouble(PyList_GetItem(listObj, i));
		if (aScore <= score)
			count++;
	}
	float percentile = count / (float)arrsz;
	return Py_BuildValue("f", percentile);
}

static PyObject* engine_percentileofscore_array(PyObject *self, PyObject *args) {
	// python: engine.percentileofscore(numpy.array[scores], score)
	double score = 0;
	PyArrayObject* arrayObj;
	if (!PyArg_ParseTuple(args, "O!d", &PyArray_Type, &arrayObj, &score))
        return NULL;
	if (not_doublevector(arrayObj)) {
		printf("invalid type!\n");
		return NULL;
	}

	// loop through python list of doubles, counting scores less than or equal to score
	double* cin = (double *)arrayObj->data;
	int arrsz = arrayObj->dimensions[0];
	if (arrsz == 0)
		return NULL;
	int i;
	int count=0;
	for (i=0; i<arrsz; i++) {
		double aScore = cin[i];
		printf("score: %lf\n", aScore);
		if (aScore <= score)
			count++;
	}
	float percentile = count / (float)arrsz;
	return Py_BuildValue("f", percentile);
}

static PyObject* engine_erf(PyObject *self, PyObject *args) {
	float x = 0;
	if (!PyArg_ParseTuple(args, "f", &x))
        return NULL;
	double sample = gsl_sf_erf(x);
	return Py_BuildValue("f", (float)sample);
}

static PyObject* engine_poisson(PyObject *self, PyObject *args) {
	float mean = 0;
	if (!PyArg_ParseTuple(args, "f", &mean))
        return NULL;
	unsigned int sample = gsl_ran_poisson(g_rng, mean);
	return Py_BuildValue("i", (int)sample);
}

static PyObject* engine_seed(PyObject *self, PyObject *args) {
	long seed = 0;
	if (!PyArg_ParseTuple(args, "l", &seed))
        return NULL;
	gsl_rng_set(g_rng, seed);
	Py_INCREF(Py_None);
	return Py_None;
}

/** debris module **/

static PyObject* debris_generate_item(PyObject* self, PyObject* args) {
	int typeid;
	float V, xord, yord, flighttime_mean, flighttime_stddev, cdav, mass_mean, mass_stddev, fa_mean, fa_stddev;
	if (!PyArg_ParseTuple(args, "fffffifffff",
			&V,
			&xord, &yord,
			&flighttime_mean,
			&flighttime_stddev,
			&typeid,
			&cdav,
			&mass_mean, &mass_stddev,
			&fa_mean, &fa_stddev))
		return NULL;

	double mass = lognormal(mass_mean, mass_stddev);
	double fa = lognormal(fa_mean, fa_stddev);
	double flight_time = lognormal(flighttime_mean, flighttime_stddev);

	double t = 9.81 * flight_time / V;
	double k = (1.2 * V * V) / (2 * 9.81 * (mass/fa) );
	double flight_distance=0;

	switch (typeid) {
	case 0:
		flight_distance = (V*V/9.81) * (1.0/k) * (0.2060*pow(k*t, 2) + 0.011*k*t);
		break;
	case 1:
		flight_distance = (V*V/9.81) * (1.0/k) * (0.072*pow(k*t, 2) + 0.3456*k*t);
		break;
	case 2:
		flight_distance = (V*V/9.81) * (1.0/k) * (0.0723*pow(k*t, 2) + 0.2376*k*t);
		break;
	}

	// Sample Impact Location
	double sigma_x = flight_distance / 3.0;
	double sigma_y = flight_distance / 12.0;
	double x, y;
	gsl_ran_bivariate_gaussian(g_rng, sigma_x, sigma_y, 1.0, &x, &y);

	// translate to footprint coords
	double item_X = xord - flight_distance + x;
	double item_Y = yord + y;

	// calculate momentum
	double pa = 1.2;
	double b = sqrt( (pa * cdav * fa) / (mass) );
	double beta_mean = 1 - exp(-b * sqrt(flight_distance));

    double dispersion = 0;
    if (beta_mean == 1.0) {
    	dispersion = 1/beta_mean + 3.0;
    	beta_mean -= 0.001;
    }
    else if (beta_mean == 0) {
    	dispersion = 4.0;
    }
    else {
        dispersion = fmax(1.0/beta_mean, 1.0/(1.0 - beta_mean)) + 3.0;
    }

    double beta_a = beta_mean * dispersion;
    double beta_b = dispersion * (1 - beta_mean);

    double sampled = gsl_ran_beta(g_rng, beta_a, beta_b);
    float item_momentum = (float)(sampled * V * mass);

	/* output: (item_momentum, item_X, item_Y)
	 */
	return Py_BuildValue("(fff)", item_momentum, item_X, item_Y);
}

/** Module setup **/

static PyMethodDef EngineMethods[] = {
	{"beta", engine_beta, METH_VARARGS, "Draw from Beta using given a,b"},
	{"lognormal", engine_lognormal, METH_VARARGS, "Draw from Lognormal using given mean, cov"},
	{"lognormal_cdf", engine_lognormal_cdf, METH_VARARGS, "lognormal cdf"},
	{"lognormal_cdf_inv", engine_lognormal_cdf_inv, METH_VARARGS, "lognormal inv cdf"},
	{"erf", engine_erf, METH_VARARGS, "erf"},
	{"bivariate_gaussian", engine_bivariate_gaussian, METH_VARARGS, "Bivariate normal distribution"},
	{"poisson", engine_poisson, METH_VARARGS, "Draw from Poisson using given mean"},
	{"percentileofscore", engine_percentileofscore, METH_VARARGS, "Return a quantile value of sorted_data"},
	{"percentileofscore_array", engine_percentileofscore_array, METH_VARARGS, "Return a quantile value of sorted_data"},
	{"seed", engine_seed, METH_VARARGS, "Set random number seed"},
	{"debris_generate_item", debris_generate_item, METH_VARARGS, "Generate debris item for source"},
	{NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initengine(void) {
	PyObject* m = Py_InitModule("engine", EngineMethods);
	if (m == NULL) 
		return;
	import_array();
	
	g_engineError = PyErr_NewException("engine.error", NULL, NULL);
	Py_INCREF(g_engineError);
	PyModule_AddObject(m, "error", g_engineError);
	
	// setup GSL random number generators
	const gsl_rng_type* T;
	gsl_rng_env_setup();
	T = gsl_rng_default;
	g_rng = gsl_rng_alloc(T);
	gsl_rng_set(g_rng, (long)time(NULL));
}

