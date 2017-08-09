/* Kernel that computes the gradient of an image, being the gradient
 * the difference between the neighbour pixels and the central pixel
 * of a cluster.
 */

__global__ void d_Gradient(float *ptrInputImage, float *ptrGradientImage, int Nx, int Ny, int Nz, int Kx, int Ky, int Kz)
{
  int i, j, k, linearIndex;
	int Kradius_x, Kradius_y, Kradius_z;
	//Kradius_x = Kx/2; Kradius_y = Ky/2; Kradius_z = Kz/2;
  float output = 0, voxelValue = 0;
  // The blocks are larger than the voxel to be processed to load the edges of the kernel
  int x = threadIdx.x + blockDim.x*blockIdx.x;
  int y = threadIdx.y + blockDim.y*blockIdx.y;
  int z = threadIdx.z + blockDim.z*blockIdx.z;

  // Check if inside the image
  if((y>=Ny)||(x>=Nx)||(z>=Nz)||(y<0)||(x<0)||(z<0))
  {
    return;
  }
	linearIndex = y + x*Ny + z*Nx*Ny; // col-wise stored matrix
  // Get the voxel value:
  voxelValue = ptrInputImage[linearIndex];
  
  // Process only the voxels inside the processing window
  if((y>=Kradius_y)&&(x>=Kradius_x)&&(z>=Kradius_z)&&(y<(Ny-Ky))&&(x<Nx-Kradius_x)&&(z<Nz-Kradius_z))
  {
    #pragma unroll
    for(i = 0; i < Kx; i++)
    {
      #pragma unroll
      for(j = 0; j < Ky; j++)
      {
				#pragma unroll
				for(k = 0; k < Kz; k++)
				{
					// Sum of differences
					linearIndex = (y-Kradius_y+j) + (x-Kradius_x+i)*Ny + (z-Kradius_z+k)*Ny*Nz;
					output += ptrInputImage[linearIndex]-voxelValue;
				}
      }
    }
    ptrGradientImage[linearIndex] = output;
  }
}

