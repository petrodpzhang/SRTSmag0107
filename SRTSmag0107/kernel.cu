#include "LBMSRT.cuh"
#include <stdlib.h>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

int main()
{
	const int Nstep = 8000;
	int savepoint = 1000;
	float time;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	LBMpara params;
	params.Nx = 40;
	params.Ny = 40;
	params.Nz = 960;
	params.rho0 = 1.0;
	params.ux0 = 0.0;
	params.uy0 = 0.0;
	params.uz0 = 0.2;

	LBMgpu lbm;

	cout << "Simulation start!" << endl;
	lbm.init(params);

	cout << "Loop start!" << endl;
	cudaEventRecord(start, 0);

	for (int step = 1; step <= Nstep; step++)
	{
		lbm.feq();
		lbm.rate_strain();
		lbm.collision();
		lbm.swap();
		lbm.boundary();		
		lbm.calrhov();

		if (step % savepoint == 0)
			lbm.output(step);
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(start);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cout << "Loop time is: " << time << endl;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	lbm.freemem();

	cudaDeviceReset();
	system("pause");
	return 0;
}