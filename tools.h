#ifndef TOOLS_HEADER
#define TOOLS_HEADER

/**
 *
 * Error handler for CUDA code
 *
 * @param err CUDA error
 * @param file File in which the error occured
 * @param line Line on which the error occured
 *
 * Example usage: ERROR_CHECK(err)
 *
 */
static inline void ErrorCheck(cudaError_t err, const char* file, int line) {
	if (err != cudaSuccess) {
		printf("Error: %s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}

#define ERROR_CHECK(err) (ErrorCheck(err, __FILE__, __LINE__))


#endif // TOOLS_HEADER