/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * filename: m-structs.h                                                   *
 *                                                                         *
 * UTIL C-source: Medical Image Conversion Utility                         *
 *                                                                         *
 * purpose      : m-structs.c header file                                  *
 *                                                                         *
 * project      : (X)MedCon by Erik Nolf                                   *
 *                                                                         *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* $Id: m-structs.h,v 1.58 2010/08/28 23:44:23 enlf Exp $
 */

/*
   Copyright (C) 1997-2010 by Erik Nolf

   This program is free software; you can redistribute it and/or modify it
   under the terms of the GNU General Public License as published by the
   Free Software Foundation; either version 2, or (at your option) any later
   version.

   This program is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
   Public License for more details.

   You should have received a copy of the GNU General Public License along
   with this program; if not, write to the Free Software Foundation, Inc.,
   59 Place - Suite 330, Boston, MA 02111-1307, USA.  */

#pragma once

/****************************************************************************
                              D E F I N E S
*****************************************************************************/

#define MDC_MAX_DIMS    8                   /* maximum number of dimensions */

/* generic  info GN */
typedef struct General_Info_t {

	char study_date[MDC_MAXSTR];
	char study_time[MDC_MAXSTR];
        char series_date[MDC_MAXSTR]; 
        char series_time[MDC_MAXSTR];
        char acquisition_date[MDC_MAXSTR];
        char acquisition_time[MDC_MAXSTR];
        char image_date[MDC_MAXSTR];
        char image_time[MDC_MAXSTR];

} GN_INFO;

/* specific info XA modality */
typedef struct Mod_XA_Info_t  {

  /* Image identification characteristics                  */
  char ImageType[MDC_MAXSTR];               /* (0008,0008) */

  /* Number of samples (color planes) should be 1          */
  Int16 Samples_Per_Pixel;                  /* (0028,0002) */

  /* Interpretation of pixel data should be MONOCROME2     */
  char Photo_Interp[MDC_MAXSTR];            /* (0028,0004) */

  /* Frame Incr Pointer (0018,1063) Time (0018,1065) Vect  */
  char Frame_Increment_Pointer[MDC_MAXSTR]; /* (0028,0009) */

  /* Relationship between pixel sample & X-ray intensity   */
  char Pixel_Intensity_Rel[MDC_MAXSTR];    /* (0028,1040)  */
  /* */
  /* kvp Peak kilo voltage output of X-Ray generator used  */
  char kvp[MDC_MAXSTR];                    /* (0018,0060)  */

  /* Radiation Setting                                     */
  char Radiation_Setting[MDC_MAXSTR];      /* (0018,1155)  */

} XA_INFO;

/* specific info MR modality */
typedef struct Mod_MR_Info_t  {

  double repetition_time;
  double echo_time;
  double inversion_time;
  double num_averages;
  double imaging_freq;
  double pixel_bandwidth;
  double flip_angle;
  double dbdt;
  Uint32 transducer_freq;
  char transducer_type[MDC_MAXSTR];
  Uint32 pulse_repetition_freq;
  char pulse_seq_name[MDC_MAXSTR];
  char steady_state_pulse_seq[MDC_MAXSTR];
  double slab_thickness;
  double sampling_freq;

} MR_INFO;

typedef struct Modality_Info_t {

  GN_INFO gn_info;
/*XA_INFO xa_info;*/
  MR_INFO mr_info;

} MOD_INFO;

/* static related data */
typedef struct Static_Data_t {

  char  label[MDC_MAXSTR];       /* label name of image    */ /* Ant/Post */
  float total_counts;            /* total counts in image  */
  float image_duration;          /* duration of image (ms) */
  Int16 start_time_hour;         /* start time hour        */
  Int16 start_time_minute;       /* start time minute      */
  Int16 start_time_second;       /* start time second      */

} STATIC_DATA;
  
/* gated SPECT related data */
typedef struct Gated_Data_t {
 
  Int8  gspect_nesting;          /* gated spect nesting    */
  float nr_projections;          /* number of projections  */
  float extent_rotation;         /* extent of rotation     */
  float study_duration;          /* study duration (ms)    */
  float image_duration;          /* image duration (ms)    */
  float time_per_proj;           /* time per proj  (ms)    */
  float window_low;              /* lower  limit   (ms)    */
  float window_high;             /* higher limit   (ms)    */
  float cycles_observed;         /* cardiac cycles observed*/
  float cycles_acquired;         /* cardiac cycles acquired*/

} GATED_DATA;

/* acquisition data */
typedef struct Acquisition_Data_t {

  Int16 rotation_direction;      /* direction of rotation  */
  Int16 detector_motion;         /* type detector motion   */
  float rotation_offset;         /* centre rotation offset */
  float radial_position;         /* radial position        */
  float angle_start;             /* start angle (interfile)*/ /* 180 - dicom */
  float angle_step;              /* angular step           */
  float scan_arc;                /* angular range          */

} ACQ_DATA;

/* dynamic data */
typedef struct Dynamic_Data_t {

  Uint32 nr_of_slices;           /* images in time frame   */
  float time_frame_start;        /* start time frame (ms)  */
  float time_frame_delay;        /* delay this frame (ms)  */
  float time_frame_duration;     /* duration   frame (ms)  */
  float delay_slices;            /* delay each slice (ms)  */

} DYNAMIC_DATA;

/* bed data */
typedef struct Bed_Data_t {

  float hoffset;                 /* horizon. position (mm) */
  float voffset;                 /* vertical position (mm) */

} BED_DATA;

/* images related data */
typedef struct Image_Data_t {

  /*             **   general data   **                    */
  Uint32 width,height;           /* image dimension        */
  Int16 bits,type;               /* bits/pixel & datatype  */
  Uint16 flags;		         /* extra flag             */
  double min, max;               /* min/max pixelvalue     */
  double qmin, qmax;             /* quantified min/max     */
  double fmin, fmax;             /* min/max in whole frame */ 
  double qfmin, qfmax;           /* in whole frame (quant) */
  float rescale_slope;           /* rescale slope          */ /* auto filled */
  float rescale_intercept;       /* rescale intercept      */ /* auto filled */
  Uint32 frame_number;           /* part of frame (1-based)*/ /* auto filled */
  float slice_start;             /* start    of slice (ms) */ /* auto filled */
  Uint8 *buf;                    /* pointer to raw image   */
  long load_location;            /* load start in file     */

  /*             **  internal items  **                    */
  Int8   rescaled;               /* rescaled image?        */ 
  double rescaled_min;           /* new rescaled max       */
  double rescaled_max;           /* new rescaled min       */ 
  double rescaled_fctr;          /* new rescaled fctr      */
  double rescaled_slope;         /* new rescaled slope     */
  double rescaled_intercept;     /* new rescaled intercept */

  /*             **   ecat64 items   **                    */
  Int16 quant_units;             /* quantification units   */
  Int16 calibr_units;            /* calibration units      */
  float quant_scale;             /* quantification scale   */
  float calibr_fctr;             /* calibration factor     */
  float intercept;               /* scale intercept        */
  float pixel_xsize;             /* pixel size X      (mm) */
  float pixel_ysize;             /* pixel size Y      (mm) */
  float slice_width;             /* slice width       (mm) */
  float recon_scale;             /* recon magnification    */

  /*            **  Acr/Nema items   **                    */

  float image_pos_dev[3];        /* image posit  dev  (mm) */
  float image_orient_dev[6];     /* image orient dev  (mm) */
  float image_pos_pat[3];        /* image posit  pat  (mm) */
  float image_orient_pat[6];     /* image orient pat  (mm) */
  float slice_spacing;           /* space btw centres (mm) */
  float ct_zoom_fctr;            /* CT image zoom factor   */

  /*            **  Miscellaneous    **                    */

  STATIC_DATA *sdata;            /* extra static entries   */ /* just one */

  unsigned char *plugb;          /* like to attach here?   */

} IMG_DATA;

/* the file information struct */
typedef struct File_Info_t {

  FILE *ifp;                     /* pointer to input file  */
  FILE *ifp_raw;                 /* pointer to raw input   */
  FILE *ofp;                     /* pointer to output file */
  FILE *ofp_raw;                 /* pointer to raw output  */ 
  char ipath[MDC_MAX_PATH+1];    /* path to input  file    */
  char opath[MDC_MAX_PATH+1];    /* path to output file    */
  char *idir;                    /* dir to input  file     */
  char *odir;                    /* dir to output file     */
  char *ifname;                  /* name of input file     */ 
  char *ofname;                  /* name of output file    */
  int  iformat;                  /* format of  input file  */
  int  oformat;                  /* format of output file  */
  int  modality;                 /* modality               */
  Int8 rawconv;                  /* FRMT_RAW | FRMT_ASCII  */
  Int8 endian;                   /* endian of file         */
  Int8 compression;              /* file compression       */
  Int8 truncated;                /* truncated file?        */
  Int8 diff_type;                /* images with diff type? */  
  Int8 diff_size;                /* images with diff size? */
  Int8 diff_scale;               /* images with diff scale?*/
  Uint32 number;                 /* total number of images */  /* private */
  Uint32 mwidth,mheight;         /* global max dimensions  */
  Int16 bits, type;              /* global bits & datatype */
  Int16 dim[MDC_MAX_DIMS];       /* [0] = # of dimensions  */
                                 /* [1] = X-dim (pixels)   */
                                 /* [2] = Y-dim (pixels)   */
                                 /* [3] = Z-dim (planes)   */
                                 /* [4] =       (frames)   */
                                 /* [5] =       (gates)    */
                                 /* [6] =       (beds)     */
                                 /* ...                    */
                                 /* values must be 1-based */

  float pixdim[MDC_MAX_DIMS];    /* [0] = # of dimensions  */
                                 /* [1] = X-dim (mm)       */
                                 /* [2] = Y-dim (mm)       */
                                 /* [3] = Z-dim (mm)       */
                                 /* [4] = time  (ms)       */
                                 /* ...                    */ 

  double glmin, glmax;           /* global min/max value   */
  double qglmin, qglmax;         /* quantified min/max     */

  Int8  contrast_remapped;       /* contrast remap applied */
  float window_centre;           /* contrast window centre */
  float window_width;            /* contrast window width  */

  Int8 slice_projection;         /* projection of images   */
  Int8 pat_slice_orient;         /* combined flag          */
  char pat_pos[MDC_MAXSTR];      /* patient position       */
  char pat_orient[MDC_MAXSTR];   /* patient orientation    */
  char  patient_sex[MDC_MAXSTR]; /* sex of patient         */
  char  patient_name[MDC_MAXSTR];/* name of patient        */
  char  patient_id[MDC_MAXSTR];  /* id   of patient        */
  char  patient_dob[MDC_MAXSTR]; /* birth of patient       */ /* YYYYMMDD */
  float patient_weight;          /* weight of patient (kg) */
  float patient_height;          /* height of patient (m)  */
  char operator_name[MDC_MAXSTR];/* name of scan operator  */
  char study_descr[MDC_MAXSTR];  /* study description      */
  char study_id[MDC_MAXSTR];     /* study id               */
  Int16 study_date_day;          /* day of study   (1-31)  */
  Int16 study_date_month;        /* month of study (1-12)  */
  Int16 study_date_year;         /* year of study          */
  Int16 study_time_hour;         /* hour of study          */
  Int16 study_time_minute;       /* minute of study        */
  Int16 study_time_second;       /* second of study        */
  Int16 dose_time_hour;          /* hour   of dose start   */
  Int16 dose_time_minute;        /* minute of dose start   */
  Int16 dose_time_second;        /* second of dose start   */
  Int32 nr_series;               /* series number          */
  Int32 nr_acquisition;          /* acquisition number     */
  Int32 nr_instance;             /* instance number        */
  Int16 acquisition_type;        /* acquisition type       */
  Int16 planar;                  /* planar or tomo  ?      */  
  Int16 decay_corrected;         /* decay corrected ?      */
  Int16 flood_corrected;         /* flood corrected ?      */

  Int16 reconstructed;           /* reconstructed ?        */
  char recon_method[MDC_MAXSTR]; /* reconstruction method  */

  char institution[MDC_MAXSTR];  /* name of institution    */
  char manufacturer[MDC_MAXSTR]; /* name of manufacturer   */
  char series_descr[MDC_MAXSTR]; /* series description     */
  char radiopharma[MDC_MAXSTR];  /* radiopharmaceutical    */
  char filter_type[MDC_MAXSTR];  /* filter type            */
  char organ_code[MDC_MAXSTR];   /* organ                  */
  char isotope_code[MDC_MAXSTR]; /* isotope                */
  float isotope_halflife;        /* isotope halflife (sec) */
  float injected_dose;           /* amount injected  (MBq) */
  float gantry_tilt;             /* gantry tilt            */

  Uint8 map;                     /* indexed 256 colormap   */
  Uint8 palette[768];            /* global palette         */
  char *comment;                 /* whatever comment       */
  Uint32 comm_length;            /* length of comment      */

  Uint32 gatednr;                /* number of gated entries*/ /* now 0 or 1 */
  GATED_DATA *gdata;             /* array of GATED_DATA    */

  Uint32 acqnr;                  /* number acq. entries    */
  ACQ_DATA *acqdata;             /* array ACQ_DATA entries */

  Uint32 dynnr;                  /* number of time frames  */
  DYNAMIC_DATA *dyndata;         /* array of DYNAMIC_DATA  */

  Uint32 bednr;                  /* number bed positions   */
  BED_DATA * beddata;            /* array of BED_DATA      */

  IMG_DATA *image;               /* array IMG_DATA images  */

  MOD_INFO *mod;                 /* modality related info  */

  unsigned char *pluga;          /* want to attach stuff?  */

} FILEINFO;

/****************************************************************************
                            F U N C T I O N S
****************************************************************************/

char *MdcCheckFI(FILEINFO *fi);
int MdcGetStructMOD(FILEINFO *fi);
int MdcGetStructID(FILEINFO *fi, Uint32 number);
int MdcGetStructSD(FILEINFO *fi, Uint32 number);
int MdcGetStructGD(FILEINFO *fi, Uint32 number);
int MdcGetStructAD(FILEINFO *fi, Uint32 number);
int MdcGetStructDD(FILEINFO *fi, Uint32 number);
int MdcGetStructBD(FILEINFO *fi, Uint32 number);
void MdcInitMOD(MOD_INFO *mod);
void MdcInitID(IMG_DATA *id);
void MdcInitSD(STATIC_DATA *sd);
void MdcInitGD(GATED_DATA *gd);
void MdcInitAD(ACQ_DATA *acq);
void MdcInitDD(DYNAMIC_DATA *dd);
void MdcInitBD(BED_DATA *bd);
void MdcInitFI(FILEINFO *fi, const char *path);
char *MdcCopyMOD(MOD_INFO *mod, MOD_INFO *src);
char *MdcCopyID(IMG_DATA *dest, IMG_DATA *src, int COPY_IMAGE);
char *MdcCopySD(STATIC_DATA *dest, STATIC_DATA *src);
char *MdcCopyGD(GATED_DATA *dest, GATED_DATA *src);
char *MdcCopyAD(ACQ_DATA *dest, ACQ_DATA *src);
char *MdcCopyDD(DYNAMIC_DATA *dest, DYNAMIC_DATA *src);
char *MdcCopyBD(BED_DATA *dst, BED_DATA *src);
char *MdcCopyFI(FILEINFO *dest, FILEINFO *src, int COPY_IMAGES, int KEEP_FILES);
void MdcFreeMODs(FILEINFO *fi);
void MdcFreeIDs(FILEINFO *fi);
void MdcFreeODs(FILEINFO *fi);
void MdcResetIDs(FILEINFO *fi);
char *MdcResetODs(FILEINFO *fi);
void MdcCleanUpFI(FILEINFO *fi);
