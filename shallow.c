#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

#if defined(_OPENMP)
#include <omp.h>
#define GET_TIME() (omp_get_wtime()) // wall time
#else
#define GET_TIME() ((double)clock() / CLOCKS_PER_SEC) // cpu time
#endif

struct parameters {
  double dx, dy, dt, max_t;
  double g, gamma;
  int source_type;
  int sampling_rate;
  char input_h_filename[256];
  char output_eta_filename[256];
  char output_u_filename[256];
  char output_v_filename[256];
};

struct data {
  int nx, ny;
  double dx, dy;
  double *values;
};

#define GET(data, i, j) ((data)->values[(data)->nx * (j) + (i)])
#define SET(data, i, j, val) ((data)->values[(data)->nx * (j) + (i)] = (val))

int read_parameters(struct parameters *param, const char *filename)
{
  FILE *fp = fopen(filename, "r");
  if(!fp) {
    printf("Error: Could not open parameter file '%s'\n", filename);
    return 1;
  }
  int ok = 1;
  if(ok) ok = (fscanf(fp, "%lf", &param->dx) == 1);
  if(ok) ok = (fscanf(fp, "%lf", &param->dy) == 1);
  if(ok) ok = (fscanf(fp, "%lf", &param->dt) == 1);
  if(ok) ok = (fscanf(fp, "%lf", &param->max_t) == 1);
  if(ok) ok = (fscanf(fp, "%lf", &param->g) == 1);
  if(ok) ok = (fscanf(fp, "%lf", &param->gamma) == 1);
  if(ok) ok = (fscanf(fp, "%d", &param->source_type) == 1);
  if(ok) ok = (fscanf(fp, "%d", &param->sampling_rate) == 1);
  if(ok) ok = (fscanf(fp, "%256s", param->input_h_filename) == 1);
  if(ok) ok = (fscanf(fp, "%256s", param->output_eta_filename) == 1);
  if(ok) ok = (fscanf(fp, "%256s", param->output_u_filename) == 1);
  if(ok) ok = (fscanf(fp, "%256s", param->output_v_filename) == 1);
  fclose(fp);
  if(!ok) {
    printf("Error: Could not read one or more parameters in '%s'\n", filename);
    return 1;
  }
  return 0;
}

void print_parameters(const struct parameters *param)
{

  printf("Parameters:\n");
  printf(" - grid spacing (dx, dy): %g m, %g m\n", param->dx, param->dy);
  printf(" - time step (dt): %g s\n", param->dt);
  printf(" - maximum time (max_t): %g s\n", param->max_t);
  printf(" - gravitational acceleration (g): %g m/s^2\n", param->g);
  printf(" - dissipation coefficient (gamma): %g 1/s\n", param->gamma);
  printf(" - source type: %d\n", param->source_type);
  printf(" - sampling rate: %d\n", param->sampling_rate);
  printf(" - input bathymetry (h) file: '%s'\n", param->input_h_filename);
  printf(" - output elevation (eta) file: '%s'\n", param->output_eta_filename);
  printf(" - output velocity (u, v) files: '%s', '%s'\n",
         param->output_u_filename, param->output_v_filename);
}

int read_data(struct data *data, const char *filename)
{
  FILE *fp = fopen(filename, "rb");
  if(!fp) {
    printf("Error: Could not open input data file '%s'\n", filename);
    return 1;
  }
  int ok = 1;
  if(ok) ok = (fread(&data->nx, sizeof(int), 1, fp) == 1);
  if(ok) ok = (fread(&data->ny, sizeof(int), 1, fp) == 1);
  if(ok) ok = (fread(&data->dx, sizeof(double), 1, fp) == 1);
  if(ok) ok = (fread(&data->dy, sizeof(double), 1, fp) == 1);
  if(ok) {
    int N = data->nx * data->ny;
    if(N <= 0) {
      printf("Error: Invalid number of data points %d\n", N);
      ok = 0;
    }
    else {
      data->values = (double*)malloc(N * sizeof(double));
      if(!data->values) {
        printf("Error: Could not allocate data (%d doubles)\n", N);
        ok = 0;
      }
      else {
        ok = (fread(data->values, sizeof(double), N, fp) == N);
      }
    }
  }
  fclose(fp);
  if(!ok) {
    printf("Error reading input data file '%s'\n", filename);
    return 1;
  }
  return 0;
}

int write_data(const struct data *data, const char *filename, int step)
{
  char out[512];
  if(step < 0)
    sprintf(out, "%s.dat", filename);
  else
    sprintf(out, "%s_%d.dat", filename, step);
  FILE *fp = fopen(out, "wb");
  if(!fp) {
    printf("Error: Could not open output data file '%s'\n", out);
    return 1;
  }
  int ok = 1;
  if(ok) ok = (fwrite(&data->nx, sizeof(int), 1, fp) == 1);
  if(ok) ok = (fwrite(&data->ny, sizeof(int), 1, fp) == 1);
  if(ok) ok = (fwrite(&data->dx, sizeof(double), 1, fp) == 1);
  if(ok) ok = (fwrite(&data->dy, sizeof(double), 1, fp) == 1);
  int N = data->nx * data->ny;
  if(ok) ok = (fwrite(data->values, sizeof(double), N, fp) == N);
  fclose(fp);
  if(!ok) {
    printf("Error writing data file '%s'\n", out);
    return 1;
  }
  return 0;
}

int write_data_vtk(const struct data *data, const char *name,
                   const char *filename, int step)
{
  char out[512];
  if(step < 0)
    sprintf(out, "%s.vti", filename);
  else
    sprintf(out, "%s_%d.vti", filename, step);

  FILE *fp = fopen(out, "wb");
  if(!fp) {
    printf("Error: Could not open output VTK file '%s'\n", out);
    return 1;
  }

  unsigned long num_points = data->nx * data->ny;
  unsigned long num_bytes = num_points * sizeof(double);

  fprintf(fp, "<?xml version=\"1.0\"?>\n");
  fprintf(fp, "<VTKFile type=\"ImageData\" version=\"1.0\" "
          "byte_order=\"LittleEndian\" header_type=\"UInt64\">\n");
  fprintf(fp, "  <ImageData WholeExtent=\"0 %d 0 %d 0 0\" "
          "Spacing=\"%lf %lf 0.0\">\n",
          data->nx - 1, data->ny - 1, data->dx, data->dy);
  fprintf(fp, "    <Piece Extent=\"0 %d 0 %d 0 0\">\n",
          data->nx - 1, data->ny - 1);

  fprintf(fp, "      <PointData Scalars=\"scalar_data\">\n");
  fprintf(fp, "        <DataArray type=\"Float64\" Name=\"%s\" "
          "format=\"appended\" offset=\"0\">\n", name);
  fprintf(fp, "        </DataArray>\n");
  fprintf(fp, "      </PointData>\n");

  fprintf(fp, "    </Piece>\n");
  fprintf(fp, "  </ImageData>\n");

  fprintf(fp, "  <AppendedData encoding=\"raw\">\n_");

  fwrite(&num_bytes, sizeof(unsigned long), 1, fp);
  fwrite(data->values, sizeof(double), num_points, fp);

  fprintf(fp, "  </AppendedData>\n");
  fprintf(fp, "</VTKFile>\n");

  fclose(fp);
  return 0;
}

int write_manifest_vtk(const char *name, const char *filename,
                       double dt, int nt, int sampling_rate)
{
  char out[512];
  sprintf(out, "%s.pvd", filename);

  FILE *fp = fopen(out, "wb");
  if(!fp) {
    printf("Error: Could not open output VTK manifest file '%s'\n", out);
    return 1;
  }

  fprintf(fp, "<VTKFile type=\"Collection\" version=\"0.1\" "
          "byte_order=\"LittleEndian\">\n");
  fprintf(fp, "  <Collection>\n");
  for(int n = 0; n < nt; n++) {
    if(sampling_rate && !(n % sampling_rate)) {
      double t = n * dt;
      fprintf(fp, "    <DataSet timestep=\"%g\" file='%s_%d.vti'/>\n", t,
              filename, n);
    }
  }
  fprintf(fp, "  </Collection>\n");
  fprintf(fp, "</VTKFile>\n");
  fclose(fp);
  return 0;
}

int init_data(struct data *data, int nx, int ny, double dx, double dy,
              double val)
{
  data->nx = nx;
  data->ny = ny;
  data->dx = dx;
  data->dy = dy;
  data->values = (double*)malloc(nx * ny * sizeof(double));
  if(!data->values){
    printf("Error: Could not allocate data\n");
    return 1;
  }
  for(int i = 0; i < nx * ny; i++) data->values[i] = val;
  return 0;
}

void free_data(struct data *data)
{
  free(data->values);
}

double interpolate_data(const struct data *data, double x, double y)
{
  // TODO: this returns the nearest neighbor, should implement actual
  // interpolation instead
  int i = (int)(x / data->dx);
  int j = (int)(y / data->dy);
  if(i < 0) i = 0;
  else if(i > data->nx - 1) i = data->nx - 1;
  if(j < 0) j = 0;
  else if(j > data->ny - 1) j = data->ny - 1;
  double val = GET(data, i, j);
  return val;
}



double interpolate_data_perso(const struct data *data, double x, double y)
{
  int i = (int)(x / data->dx);
  int j = (int)(y / data->dy);
  if(i < 0) i = 0;
  else if(i > data->nx - 2) i = data->nx - 2;  //"-2" for still being in the grid while looking for the value above
  if(j < 0) j = 0;
  else if(j > data->ny - 2) j = data->ny - 2;

  // Calculate the positions of the surrounding grid points
  double x_coord = i * data->dx;
  double x1_coord = (i + 1) * data->dx;
  double y_coord = j * data->dy;
  double y1_coord = (j + 1) * data->dy;  
  
  //get the values around the given point /test
  double i_j_val  = GET(data, i, j);
  double i1_j_val  = GET(data, i+1, j);
  double i_j1_val  = GET(data, i, j+1);
  double i1_j1_val  = GET(data, i+1, j+1);
  
  //given the formula for bilinear interpolation
  double interpolated_value = 
      (i_j_val * (x1_coord - x) * (y1_coord - y) +
       i1_j_val * (x - x_coord) * (y1_coord - y) +
       i_j1_val * (x1_coord - x) * (y - y_coord) +
       i1_j1_val * (x - x_coord) * (y - y_coord))
      / (data->dx * data->dy);
  return interpolated_value;
}


int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv); // Initialize MPI

  int world_size, rank; // Number of processes, rank of the process
  MPI_Comm_size(MPI_COMM_WORLD, &world_size); // Get the number of processes
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank of the process

  // Create Cartesian topology
  int dims[2] = {0, 0}; // Initialize dimensions to 0
  MPI_Dims_create(world_size, 2, dims); // Create dimensions for 2D grid

  int periods[2] = {0, 0}; // Non-periodic grid
  int reorder = 0; // No reordering of processes
  MPI_Comm cart_comm; // Cartesian communicator
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart_comm);

  // Get Cartesian rank and coordinates
  int cart_rank;
  int coords[2];
  MPI_Comm_rank(cart_comm, &cart_rank);
  MPI_Cart_coords(cart_comm, cart_rank, 2, coords);

  printf("Rank = %d, Cartesian Rank = %d, Coords = (%d, %d)\n", rank, cart_rank, coords[0], coords[1]);

  if(argc != 2) {
    printf("Usage: %s parameter_file\n", argv[0]);
    MPI_Finalize();
    return 1;
  }

  struct parameters param;
  if(read_parameters(&param, argv[1])) {
    MPI_Finalize();
    return 1;
  }
  if(rank == 0){
    print_parameters(&param);
  } 
  

  struct data h;
  if(read_data(&h, param.input_h_filename)) {
    MPI_Finalize();
    return 1;
  }

  // Infer size of domain from input elevation data
  double hx = h.nx * h.dx;
  double hy = h.ny * h.dy;
  int nx = floor(hx / param.dx);
  int ny = floor(hy / param.dy);
  if(nx <= 0) nx = 1;
  if(ny <= 0) ny = 1;
  int nt = floor(param.max_t / param.dt);

  if (rank == 0) {
    printf(" - grid size: %g m x %g m (%d x %d = %d grid points)\n", hx, hy, nx, ny, nx * ny);
    printf(" - number of time steps: %d\n", nt);
  }
  

  // Calculate local domain sizes for each node
  int local_nx = nx / dims[0] + (coords[0] < nx % dims[0]);
  int local_ny = ny / dims[1] + (coords[1] < ny % dims[1]);

  struct data local_eta, local_u, local_v;
  init_data(&local_eta, local_nx, local_ny, param.dx, param.dx, 0.);
  init_data(&local_u, local_nx + 1, local_ny, param.dx, param.dy, 0.);
  init_data(&local_v, local_nx, local_ny + 1, param.dx, param.dy, 0.);

  // Interpolate bathymetry
  struct data local_h_interp_u, local_h_interp_v;
  init_data(&local_h_interp_u, local_nx + 1, local_ny, param.dx, param.dy, 0.);
  init_data(&local_h_interp_v, local_nx, local_ny + 1, param.dx, param.dy, 0.);

  for(int j = 0; j < local_ny; j++) {
    for(int i = 0; i < local_nx + 1; i++) {
      double x = (coords[0] * local_nx + i) * param.dx;
      double y = (coords[1] * local_ny + j + 0.5) * param.dy;
      double h_u = interpolate_data_perso(&h, x, y);
      SET(&local_h_interp_u, i, j, h_u);
    }
  }

  for(int j = 0; j < local_ny + 1; j++) {
    for(int i = 0; i < local_nx; i++) {
      double x = (coords[0] * local_nx + i + 0.5) * param.dx;
      double y = (coords[1] * local_ny + j) * param.dy;
      double h_v = interpolate_data_perso(&h, x, y);
      SET(&local_h_interp_v, i, j, h_v);
    }
  }



  double start = GET_TIME();

  int up, down, left, right;
  MPI_Cart_shift(cart_comm, 0, 1, &left, &right);
  MPI_Cart_shift(cart_comm, 1, 1, &down, &up);

  double *buffer_send_right = (right != MPI_PROC_NULL ? malloc(sizeof(double) * local_ny) : NULL);
  double *buffer_send_up = (up != MPI_PROC_NULL ? malloc(sizeof(double) * local_nx) : NULL);
  double *buffer_send_left_u = (left != MPI_PROC_NULL ? malloc(sizeof(double) * local_ny) : NULL);
  double *buffer_send_down_v = (down != MPI_PROC_NULL ? malloc(sizeof(double) * local_nx) : NULL);
  double *buffer_recv_left = (left != MPI_PROC_NULL ? malloc(sizeof(double) * local_ny) : NULL);
  double *buffer_recv_down = (down != MPI_PROC_NULL ? malloc(sizeof(double) * local_nx) : NULL);
  double *buffer_recv_right_u = (right != MPI_PROC_NULL ? malloc(sizeof(double) * local_ny) : NULL);
  double *buffer_recv_up_v = (up != MPI_PROC_NULL ? malloc(sizeof(double) * local_nx) : NULL);

  if(((!buffer_send_right || !buffer_recv_right_u) && right != MPI_PROC_NULL) ||
     ((!buffer_send_up || !buffer_recv_up_v) && up != MPI_PROC_NULL) ||
     ((!buffer_recv_down || !buffer_send_down_v) && down != MPI_PROC_NULL) ||
     ((!buffer_recv_left || !buffer_send_left_u) && left != MPI_PROC_NULL)) {
    printf("Error: Could not initiate buffers\n");
    free(buffer_recv_down);
    free(buffer_recv_left);
    free(buffer_recv_up_v);
    free(buffer_recv_right_u);
    free(buffer_send_down_v);
    free(buffer_send_left_u);
    free(buffer_send_right);
    free(buffer_send_up);
    MPI_Finalize();
    return 1;
  }



  for(int n = 0; n < nt; n++) {
    if(n && (n % (nt / 10)) == 0) {
      double time_sofar = GET_TIME() - start;
      double eta = (nt - n) * time_sofar / n;
      printf("Computing step %d/%d (ETA: %g seconds)     \r", n, nt, eta);
      fflush(stdout);
    }

    if (param.sampling_rate && !(n % param.sampling_rate)) {
        // Allocate memory for the gathered data on the root process
        double *global_eta = NULL;
        int global_nx = nx * ny; // Total number of grid points
        if (rank == 0) {
            global_eta = (double *)malloc(global_nx * sizeof(double));
            if (!global_eta) {
                printf("Error: Could not allocate memory for global_eta\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
      
  
      // Gather local_eta data from all processes to the root process
      MPI_Gather(local_eta.values, local_nx * local_ny, MPI_DOUBLE,
                  global_eta, local_nx * local_ny, MPI_DOUBLE,
                  0, MPI_COMM_WORLD);

      // Write the gathered data to a VTK file on the root process
      if (rank == 0) {
          struct data global_eta_data;
          global_eta_data.nx = nx;
          global_eta_data.ny = ny;
          global_eta_data.dx = param.dx;
          global_eta_data.dy = param.dy;
          global_eta_data.values = global_eta;
  
          write_data_vtk(&global_eta_data, "water elevation", param.output_eta_filename, n);
  
          free(global_eta);
      }
    }

    double t = n * param.dt;

    if(param.source_type == 1) {
      if (coords[1] == 0) {
        double A = 5;
        double f = 1. / 20.;
        for(int i = 0; i < local_nx; i++) {
          SET(&local_v, i, local_ny - 1, A * sin(2 * M_PI * f * t));
          SET(&local_v, i, 0, 0.);
        }
        for(int j = 0; j < local_ny; j++) {
          SET(&local_u, 0, j, 0.);  
          SET(&local_u, local_nx - 1, j, 0.);
        }
      }
    } else if(param.source_type == 2 &&
              (coords[0] * local_nx <= nx / 2 && (coords[0] + 1) * local_nx > nx / 2 &&
               coords[1] * local_ny <= ny / 2 && (coords[1] + 1) * local_ny > ny / 2)) {
      double A = 5;
      double f = 1. / 20.;
      int local_i = nx / 2 - coords[0] * local_nx;
      int local_j = ny / 2 - coords[1] * local_ny;
      SET(&local_eta, local_i, local_j, A * sin(2 * M_PI * f * t));
    } else {
      printf("Error: Unknown source type %d\n", param.source_type);
      MPI_Finalize();
      return 1;
    }

    for(int i = 0; i < local_nx; i++) {
      if(down != MPI_PROC_NULL)
        buffer_send_down_v[i] = GET(&local_v, i, 0);
      if(up != MPI_PROC_NULL)
        buffer_send_up[i] = GET(&local_eta, i, local_ny - 1);
    }
    for(int j = 0; j < local_ny; j++) {
      if(left != MPI_PROC_NULL)
        buffer_send_left_u[j] = GET(&local_u, 0, j);
      if(right != MPI_PROC_NULL)
        buffer_send_right[j] = GET(&local_eta, local_nx - 1, j);
    }
  
    MPI_Request requests[8];

     if (up != MPI_PROC_NULL) {
        MPI_Isend(buffer_send_up, local_nx, MPI_DOUBLE, up, 0, cart_comm, &requests[0]);       
        MPI_Irecv(buffer_recv_up_v, local_nx, MPI_DOUBLE, up, 1, cart_comm, &requests[4]);
    } else {
        requests[0] = MPI_REQUEST_NULL;
        requests[4] = MPI_REQUEST_NULL;
    }
    
    if (down != MPI_PROC_NULL) {
        MPI_Isend(buffer_send_down_v, local_nx, MPI_DOUBLE, down, 1, cart_comm, &requests[1]);
        MPI_Irecv(buffer_recv_down, local_nx, MPI_DOUBLE, down, 0, cart_comm, &requests[5]);
    } else {
        requests[1] = MPI_REQUEST_NULL;
        requests[5] = MPI_REQUEST_NULL;
    }
    
    if (left != MPI_PROC_NULL) {
        MPI_Isend(buffer_send_left_u, local_ny, MPI_DOUBLE, left, 2, cart_comm, &requests[2]);
        MPI_Irecv(buffer_recv_left, local_ny, MPI_DOUBLE, left, 3, cart_comm, &requests[6]);
    } else {
        requests[2] = MPI_REQUEST_NULL;
        requests[6] = MPI_REQUEST_NULL;
    }
    
    if (right != MPI_PROC_NULL) {
        MPI_Isend(buffer_send_right, local_ny, MPI_DOUBLE, right, 3, cart_comm, &requests[3]);
        MPI_Irecv(buffer_recv_right_u, local_ny, MPI_DOUBLE, right, 2, cart_comm, &requests[7]);
    } else {
        requests[3] = MPI_REQUEST_NULL;
        requests[7] = MPI_REQUEST_NULL;
    }


    MPI_Waitall(8, requests, MPI_STATUSES_IGNORE);

    for(int j = 0; j < local_ny; j++) {
      for(int i = 0; i < local_nx; i++) {
        double u_1 = (i == local_nx - 1 && right != MPI_PROC_NULL) ? buffer_recv_right_u[j] : GET(&local_u, i + 1, j);
        double v_1 = (j == local_ny - 1 && up != MPI_PROC_NULL) ? buffer_recv_up_v[i] : GET(&local_v, i, j + 1);

        double eta_ij = GET(&local_eta, i, j)
          - param.dt / param.dx * (GET(&local_h_interp_u, i + 1, j) * u_1 - GET(&local_h_interp_u, i, j) * GET(&local_u, i, j))
          - param.dt / param.dy * (GET(&local_h_interp_v, i, j + 1) * v_1 - GET(&local_h_interp_v, i, j) * GET(&local_v, i, j));
        SET(&local_eta, i, j, eta_ij);
      }
    }

    for(int j = 0; j < local_ny; j++) {
      for(int i = 0; i < local_nx; i++) {
        double c1 = param.dt * param.g;
        double c2 = param.dt * param.gamma;
        double eta_ij = GET(&local_eta, i, j);
        double eta_imj = (coords[0] * local_nx == 0) ? GET(&local_eta, (i == 0) ? 0 : i - 1, j) : buffer_recv_left[j];
        double eta_ijm = (coords[1] * local_ny == 0) ? GET(&local_eta, i, (j == 0) ? 0 : j - 1) : buffer_recv_down[i];
        double u_ij = (1. - c2) * GET(&local_u, i, j)
          - c1 / param.dx * (eta_ij - eta_imj);
        double v_ij = (1. - c2) * GET(&local_v, i, j)
          - c1 / param.dy * (eta_ij - eta_ijm);
        SET(&local_u, i, j, u_ij);
        SET(&local_v, i, j, v_ij);
      }
    }
  }
  


  double time = GET_TIME() - start;
  printf("\nDone: %g seconds (%g MUpdates/s)\n", time, 1e-6 * (double)local_eta.nx * (double)local_eta.ny * (double)nt / time);

  // Fill buffers
  for(int i = 0; i < local_nx; i++){
    if(down != MPI_PROC_NULL)
      buffer_send_down_v[i] = GET(&local_v,i,0);
    if(up != MPI_PROC_NULL )
      buffer_send_up[i] = GET(&local_eta,i,local_ny-1);
  }
  for(int j = 0 ; j < local_ny; j++){
    if(left != MPI_PROC_NULL)
      buffer_send_left_u[j] = GET(&local_u,0,j);
    if(right != MPI_PROC_NULL)
    buffer_send_right[j] = GET(&local_eta,local_nx-1,j);
  }



  //write_manifest_vtk("water elevation", param.output_eta_filename,
  //                   param.dt, nt, param.sampling_rate);
  //write_manifest_vtk("x velocity", param.output_u_filename,
  //                   param.dt, nt, param.sampling_rate);
  //write_manifest_vtk("y velocity", param.output_v_filename,
  //                   param.dt, nt, param.sampling_rate);

  /*
  double time = GET_TIME() - start;
  printf("\nDone: %g seconds (%g MUpdates/s)\n", time,
         1e-6 * (double)eta.nx * (double)eta.ny * (double)nt / time);
  */
   if (rank == 0) {
    write_manifest_vtk("water elevation", param.output_eta_filename, param.dt, nt, param.sampling_rate);
  }

  free_data(&local_h_interp_u);
  free_data(&local_h_interp_v);
  free_data(&local_eta);
  free_data(&local_u);
  free_data(&local_v);

  free(buffer_recv_down);
  free(buffer_recv_left);
  free(buffer_recv_up_v);
  free(buffer_recv_right_u);
  free(buffer_send_down_v);
  free(buffer_send_left_u);
  free(buffer_send_right);
  free(buffer_send_up);

  MPI_Finalize(); // Finalize MPI

  return 0;
}

