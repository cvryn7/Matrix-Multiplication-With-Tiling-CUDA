/**
*Developed By Karan Bhagat
*March 2017
**/

#include <stdio.h>
#include <stdlib.h>

//cuda kernel for multiplying two matrices without tiling
__global__ void matrix_mul_kernel(int* a, int* b, int* c, int a_rows, int a_columns, int b_columns)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	
	//check if thread directly maps to the dimensions of resulting matrix
	if (row < a_rows && col < b_columns)
	{
		int result = 0;
		int k;
		for (k = 0; k < a_columns; k++)
		{
			result += (a[row * a_columns + k] * b[k * b_columns + col]);
		}
		c[row * b_columns + col] = result;
	}
}

void build_matrix(FILE *file, int* mat, int rows, int columns);

int main(int argc, char **argv)
{
	//check for filenames and matrices' dimensions
	if (argc != 6)
	{
		printf("Usage : ./matrix_mul_tiling <fileA> <fileB> <A_rows> <A_columns> <B_columns>");
		exit(1);
	}
	
	char* fileA_name = argv[1];//matrix A filename
	char* fileB_name = argv[2];//matrix B filename

	// a_columns can also be perceived as b_rows
	int a_rows, a_columns, b_columns;
	
	//read matrix A and B's dimensions
	sscanf(argv[3], "%d", &a_rows);
	sscanf(argv[4], "%d", &a_columns);
	sscanf(argv[5], "%d", &b_columns);
	
	FILE *fileA = fopen(fileA_name, "r");
	FILE *fileB = fopen(fileB_name, "r");

	//declare host and device matrices pointers
	int* mat_a;
	int* mat_b;
	int* mat_c;
	int* d_mat_a;
	int* d_mat_b;
	int* d_mat_c;
	
	//allocate memory for host matrices
	mat_a = (int*)malloc(a_rows * a_columns * sizeof(int));
	mat_b = (int*)malloc(a_columns * b_columns * sizeof(int));
	mat_c = (int*)malloc(a_rows * b_columns * sizeof(int));
	
	int i, j;
	
	build_matrix(fileA, mat_a, a_rows, a_columns);
	build_matrix(fileB, mat_b, a_columns, b_columns);
	
	//declare dimensions for the grid and block
	dim3 dimBlock(2,2);
	dim3 dimGrid((int)ceil(b_columns/2),(int)ceil(a_rows/2));
	
	const size_t size_a = a_rows * a_columns * sizeof(int);
	const size_t size_b = a_columns * b_columns * sizeof(int);
	const size_t size_c = a_rows * b_columns * sizeof(int);

	//allocate matrices memeory on device
	cudaMalloc((void **)&d_mat_a, size_a);
	cudaMalloc((void **)&d_mat_b, size_b);
	cudaMalloc((void **)&d_mat_c, size_c);

	//copy A and B matrices from host to device
	cudaMemcpy(d_mat_a, mat_a, size_a, cudaMemcpyHostToDevice);
	cudaMemcpy(d_mat_b, mat_b, size_b, cudaMemcpyHostToDevice);

	//execute cuda kernel
	matrix_mul_kernel<<<dimGrid, dimBlock>>>(d_mat_a, d_mat_b, d_mat_c, a_rows, a_columns, b_columns);

	//copy the compute matrix C from device to host
	cudaMemcpy(mat_c, d_mat_c, size_c, cudaMemcpyDeviceToHost);
	
	//free cuda memory
	cudaFree(d_mat_a);
	cudaFree(d_mat_b);
	cudaFree(d_mat_c);

	//print the resulting matrix
	for (i = 0; i < a_rows; i++)
	{
		for (j = 0; j < b_columns; j++)
		{
			printf("%d ", mat_c[i * b_columns + j]);
		}
		printf("\n");
	}
}

//build matrix from the file
void build_matrix(FILE *file, int* mat, int rows, int columns)
{
	int i, j;
	for (i = 0; i < rows; i++)
	{
		for (j = 0; j < columns; j++) 
		{
			fscanf(file, "%d", &mat[i * columns + j]);
		}
	}
}