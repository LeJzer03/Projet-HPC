#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <time.h>

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

double bilinear_interpolation_with_edge_handling(const struct data *data, double x, double y) 
{
    // Find the indices of the surrounding grid points
    int i = (int)(x / data->dx);  // Grid point to the left
    int j = (int)(y / data->dy);  // Grid point below

    // Ensure indices are within bounds, but stop at second to last point
    if (i < 0) i = 0;
    if (i > data->nx - 2) i = data->nx - 2; // Stop at second to last to have right neighbor
    if (j < 0) j = 0;
    if (j > data->ny - 2) j = data->ny - 2; // Stop at second to last to have top neighbor

    // Calculate the positions of the surrounding grid points
    double x1 = i * data->dx;
    double x2 = (i + 1) * data->dx;
    double y1 = j * data->dy;
    double y2 = (j + 1) * data->dy;

    // Check if point is on the edges or inside the grid
    int on_left_edge = (i == 0);
    int on_right_edge = (i == data->nx - 2);
    int on_bottom_edge = (j == 0);
    int on_top_edge = (j == data->ny - 2);

    // Retrieve the available values
    double fP1 = GET(data, i, j);         // Bottom-left (P1)
    double fP2 = GET(data, i + 1, j);     // Bottom-right (P2)
    double fP3 = GET(data, i, j + 1);     // Top-left (P3)
    double fP4 = GET(data, i + 1, j + 1); // Top-right (P4)

    // Handle cases where point is on an edge
    double denom = (x2 - x1) * (y2 - y1);  // Denominator for bilinear interpolation

    // Interpolation value
    double fxy;

    // Case 1: On a corner
    if (on_left_edge && on_bottom_edge) {
        fxy = (fP1 + fP2 + fP3) / 3.0;  // Average of three points (P1, P2, P3)
    } else if (on_right_edge && on_bottom_edge) {
        fxy = (fP1 + fP2 + fP4) / 3.0;  // Average of three points (P1, P2, P4)
    } else if (on_left_edge && on_top_edge) {
        fxy = (fP1 + fP3 + fP4) / 3.0;  // Average of three points (P1, P3, P4)
    } else if (on_right_edge && on_top_edge) {
        fxy = (fP2 + fP3 + fP4) / 3.0;  // Average of three points (P2, P3, P4)
    }
    // Case 2: On an edge
    else if (on_left_edge) {
        fxy = (fP1 * (y2 - y) + fP3 * (y - y1)) / (y2 - y1);  // Linear interpolation vertically
    } else if (on_right_edge) {
        fxy = (fP2 * (y2 - y) + fP4 * (y - y1)) / (y2 - y1);  // Linear interpolation vertically
    } else if (on_bottom_edge) {
        fxy = (fP1 * (x2 - x) + fP2 * (x - x1)) / (x2 - x1);  // Linear interpolation horizontally
    } else if (on_top_edge) {
        fxy = (fP3 * (x2 - x) + fP4 * (x - x1)) / (x2 - x1);  // Linear interpolation horizontally
    }
    // Case 3: Inside the grid (regular bilinear interpolation)
    else {
        fxy = (fP1 * (x2 - x) * (y2 - y) +
               fP2 * (x - x1) * (y2 - y) +
               fP3 * (x2 - x) * (y - y1) +
               fP4 * (x - x1) * (y - y1)) / denom;
    }

    return fxy;
}

/***
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
***/

int main(int argc, char **argv)
{
  if(argc != 2) {
    printf("Usage: %s parameter_file\n", argv[0]);
    return 1;
  }

  struct parameters param;
  if(read_parameters(&param, argv[1])) return 1;
  print_parameters(&param);

  struct data h;
  if(read_data(&h, param.input_h_filename)) return 1;

  // infer size of domain from input elevation data
  double hx = h.nx * h.dx;
  double hy = h.ny * h.dy;
  int nx = floor(hx / param.dx);
  int ny = floor(hy / param.dy);
  if(nx <= 0) nx = 1;
  if(ny <= 0) ny = 1;
  int nt = floor(param.max_t / param.dt);

  printf(" - grid size: %g m x %g m (%d x %d = %d grid points)\n",
         hx, hy, nx, ny, nx * ny);
  printf(" - number of time steps: %d\n", nt);

  struct data eta, u, v;
  init_data(&eta, nx, ny, param.dx, param.dx, 0.);
  init_data(&u, nx + 1, ny, param.dx, param.dy, 0.);
  init_data(&v, nx, ny + 1, param.dx, param.dy, 0.);

  // interpolate bathymetry

  /* struct data h_interp;
  init_data(&h_interp, nx, ny, param.dx, param.dy, 0.);
  for(int i = 0 ; i < nx ; i++){
  for(int j = 0 ; j < ny ; j++){
  double x = i*param.dx;
  double y = j*param.dy;
  double val = interpolate_data(&h,x,y);
  SET(&h_interp,i,j);
  }
  }
  */ 
  

  double start = GET_TIME();

  for(int n = 0; n < nt; n++) {

    if(n && (n % (nt / 10)) == 0) {
      double time_sofar = GET_TIME() - start;
      double eta = (nt - n) * time_sofar / n;
      printf("Computing step %d/%d (ETA: %g seconds)     \r", n, nt, eta);
      fflush(stdout);
    }

    // output solution
    if(param.sampling_rate && !(n % param.sampling_rate)) {
      write_data_vtk(&eta, "water elevation", param.output_eta_filename, n);
      //write_data_vtk(&u, "x velocity", param.output_u_filename, n);
      //write_data_vtk(&v, "y velocity", param.output_v_filename, n);
    }

    // impose boundary conditions
    double t = n * param.dt;
    if(param.source_type == 1) {
      // sinusoidal velocity on top boundary
      double A = 5;
      double f = 1. / 20.;
      for(int i = 0; i < nx; i++) {
        for(int j = 0; j < ny; j++) {
          SET(&u, 0, j, 0.);
          SET(&u, nx, j, 0.);
          SET(&v, i, 0, 0.);
          SET(&v, i, ny, A * sin(2 * M_PI * f * t));
        }
      }
    }
    else if(param.source_type == 2) {
      // sinusoidal elevation in the middle of the domain
      double A = 5;
      double f = 1. / 20.;
      SET(&eta, nx / 2, ny / 2, A * sin(2 * M_PI * f * t));
    }
    else {
      // TODO: add other sources
      printf("Error: Unknown source type %d\n", param.source_type);
      exit(0);
    }

    // update eta
    for(int i = 0; i < nx; i++) {
      for(int j = 0; j < ny ; j++) {
        // TODO: this does not evaluate h at the correct locations
      double xu = i * param.dx;
      double xu1 = (i+1) * param.dx;
      double yu = (j+1/2) * param.dy;
      double xv = (i+1/2) * param.dx;
      double yv = j * param.dy;
      double yv1 = (j+1)*param.dy;
      //double val = interpolate_data(&h, x, y);   //bilinear_interpolation_with_edge_handling
      double h_u1 = bilinear_interpolation_with_edge_handling(&h, xu1, yu);
      double h_v1 = bilinear_interpolation_with_edge_handling(&h, xv, yv1);
      double h_v = bilinear_interpolation_with_edge_handling(&h, xv, yv);
      double h_u = bilinear_interpolation_with_edge_handling(&h, xu, yu);
        double eta_ij = GET(&eta, i, j)
          - param.dt / param.dx * (h_u1*GET(&u, i + 1, j) - h_u*GET(&u, i, j))
          - param.dt / param.dy * (h_v1*GET(&v, i, j + 1) - h_v*GET(&v, i, j));
        SET(&eta, i, j, eta_ij);
      }
    }

    // update u and v
    for(int i = 0; i < nx; i++) {
      for(int j = 0; j < ny; j++) {
        double c1 = param.dt * param.g;
        double c2 = param.dt * param.gamma;
        double eta_ij = GET(&eta, i, j);
        double eta_imj = GET(&eta, (i == 0) ? 0 : i - 1, j);
        double eta_ijm = GET(&eta, i, (j == 0) ? 0 : j - 1);
        double u_ij = (1. - c2) * GET(&u, i, j)
          - c1 / param.dx * (eta_ij - eta_imj);
        double v_ij = (1. - c2) * GET(&v, i, j)
          - c1 / param.dy * (eta_ij - eta_ijm);
        SET(&u, i, j, u_ij);
        SET(&v, i, j, v_ij);
      }
    }

  }

  write_manifest_vtk("water elevation", param.output_eta_filename,
                     param.dt, nt, param.sampling_rate);
  //write_manifest_vtk("x velocity", param.output_u_filename,
  //                   param.dt, nt, param.sampling_rate);
  //write_manifest_vtk("y velocity", param.output_v_filename,
  //                   param.dt, nt, param.sampling_rate);

  double time = GET_TIME() - start;
  printf("\nDone: %g seconds (%g MUpdates/s)\n", time,
         1e-6 * (double)eta.nx * (double)eta.ny * (double)nt / time);

  //free_data(&h_interp);
  free_data(&eta);
  free_data(&u);
  free_data(&v);

  return 0;
}
