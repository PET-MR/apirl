/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * filename: m-structs.c                                                   *
 *                                                                         *
 * UTIL C-source: Medical Image Conversion Utility                         *
 *                                                                         *
 * purpose      : structs handling functions                               *
 *                                                                         *
 * project      : (X)MedCon by Erik Nolf                                   *
 *                                                                         *
 * Functions    : MdcCheckFI()          - Check FILEINFO struct integrity  * 
 *                MdcGetStructMOD()     - Get MOD_INFO structs             *
 *                MdcGetStructID()      - Get IMG_DATA structs             *
 *                MdcGetStructSD()      - Get STATIC_DATA structs          *
 *                MdcGetStructGD()      - Get GATED_DATA structs           *
 *                MdcGetStructAD()      - Get ACQ_DATA structs             *
 *                MdcGetStructDD()      - Get DYNAMIC_DATA structs         *
 *                MdcGetStructBD()      - Get BED_DATA structs             *
 *                MdcInitMOD()          - Initialize MOD_INFO struct       *
 *                MdcInitID()           - Initialize IMG_DATA structs      *
 *                MdcInitSD()           - Initialize STATIC_DATA strucs    *
 *                MdcInitGD()           - Initialize GATED_DATA structs    *
 *                MdcInitAD()           - Initialize ACQ_DATA structs      *
 *                MdcInitDD()           - Initialize DYNAMIC_DATA structs  *
 *                MdcInitBD()           - Initialize BED_DATA structs      *
 *                MdcInitFI()           - Initialize FILEINFO struct       *
 *                MdcCopyID()           - Copy IMG_DATA     information    *
 *                MdcCopySD()           - Copy STATIC_DATA  information    *
 *                MdcCopyGD()           - Copy GATED_DATA   information    *
 *                MdcCopyAD()           - Copy ACQ_DATA     information    *
 *                MdcCopyDD()           - Copy DYNAMIC_DATA information    *
 *                MdcCopyBD()           - Copy BED_DATA     information    *
 *                MdcCopyFI()           - Copy FILEINFO     information    *
 *                MdcFreeIDs()          - Free IMG_DATA structs            *
 *                MdcFreeMODs()         - Free MOD_INFO structs            *
 *                MdcResetIDs()         - Reset IMG_DATA structs           *
 *                MdcResetODs()         - Reset all others except IMG_DATA *
 *                MdcCleanUpFI()        - Clean up FILEINFO struct         *
 *                                                                         *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* $Id: m-structs.c,v 1.79 2010/08/28 23:44:23 enlf Exp $ 
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
 
/****************************************************************************
                              H E A D E R S
****************************************************************************/

#include "m-depend.h"

#include <stdio.h>
#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif
#ifdef HAVE_STRING_H
#include <string.h>
#endif
#ifdef HAVE_STRINGS_H
#ifndef _WIN32
#include <strings.h>
#endif
#endif

#include "medcon.h"

/****************************************************************************
                            F U N C T I O N S
****************************************************************************/

char *MdcCheckFI(FILEINFO *fi)
{
  Uint32 i, t;

  /* check fi->dim[] values */ 
  if (fi->dim[0] <= 2 ) {
    sprintf(mdcbufr,"Internal ## fi->dim[0]=%d",fi->dim[0]);
    return(mdcbufr);
  }else{
    for (i=1; i<=fi->dim[0]; i++) {
       if (fi->dim[i] <= 0 ) {
         sprintf(mdcbufr,"Internal ## fi->dim[%d]=%d",i,fi->dim[i]);
         return(mdcbufr);
       }
    }
  }

  /* all fi->dim[] are 1-based, even unused */
  for (i=0; i<MDC_MAX_DIMS; i++) if (fi->dim[i] <= 0)
     return("Internal ## Dangerous negative fi->dim values");


  /* check fi->number value */
  for (i=1, t=3; t <= fi->dim[0]; t++) {
     i*=fi->dim[t];
  }
  if (fi->number != i) return("Internal ## Improper fi->dim values");

  return(NULL);

}

/* returns 1 on success and 0 on failure */
int  MdcGetStructMOD(FILEINFO *fi)
{
   fi->mod = calloc(sizeof(MOD_INFO),1);

   if (fi->mod == NULL) return(MDC_NO);

   return(MDC_YES);

}

/* returns 1 on success and 0 on failure */
int  MdcGetStructID(FILEINFO *fi, Uint32 number)
{
  Uint32 i, begin=number; 

  /* bad request */
  if (number == 0) return(MDC_NO);

  /* allocate structs */
  if (fi->image == NULL) {
    /* fresh allocation */
    fi->image=(IMG_DATA *)malloc(sizeof(IMG_DATA)*number); 
    begin = 0; 
  }else if (number != fi->number) {
    /* reallocation */
    fi->image=(IMG_DATA *)realloc(fi->image,sizeof(IMG_DATA)*number);
    begin = number > fi->number ? fi->number : number;
  }

  if (fi->image == NULL) { fi->number=0; return(MDC_NO); }

  /* initialize new structs */
  for (i=begin; i<number; i++) MdcInitID(&fi->image[i]);  

  /* set new number */
  fi->number = number;

  return(MDC_YES);
}

/* returns 1 on success and 0 on failure        */
int MdcGetStructSD(FILEINFO *fi, Uint32 number)
{
  STATIC_DATA *sdata;
  Uint32 i;

  if (number != fi->number) return(MDC_NO); /* always for number of images */

  for (i=0; i<fi->number; i++) {
     sdata = (STATIC_DATA *)malloc(sizeof(STATIC_DATA));
     if (sdata == NULL) return(MDC_NO);
     MdcInitSD(sdata);
     fi->image[i].sdata = sdata;
  }

  return(MDC_YES);
}

/* returns 1 on success and 0 on failure        */
int MdcGetStructGD(FILEINFO *fi, Uint32 number)
{
  Uint32 i, begin=number;

  if (number == 0) return(MDC_NO);

  if (fi->gdata == NULL) {
    /* fresh allocation */
    fi->gdata = (GATED_DATA *)malloc(sizeof(GATED_DATA)*number);
    begin = 0;
  }else if (number != fi->gatednr) {
    /* reallocation */
    fi->gdata = (GATED_DATA *)realloc(fi->gdata,sizeof(GATED_DATA)*number);
    begin = number > fi->gatednr ? fi->gatednr : number;
  }

  if (fi->gdata == NULL) { fi->gatednr=0; return(MDC_NO); }

  /* initialize new structs */
  for (i=begin; i<number; i++) MdcInitGD(&fi->gdata[i]);

  /* set new number */
  fi->gatednr = number;

  return(MDC_YES);
}

/* returns 1 on success and 0 on failure        */
int MdcGetStructAD(FILEINFO *fi, Uint32 number)
{
  Uint32 i, begin=number;

  if (number == 0) return(MDC_NO);

  if (fi->acqdata == NULL) {
    /* fresh allocation */
    fi->acqdata = (ACQ_DATA *)malloc(sizeof(ACQ_DATA)*number);
    begin = 0;
  }else if (number != fi->acqnr) {
    /* reallocation */
    fi->acqdata = (ACQ_DATA *)realloc(fi->acqdata,sizeof(ACQ_DATA)*number);
    begin = number > fi->acqnr ? fi->acqnr : number;
  }

  if (fi->acqdata == NULL) { fi->acqnr=0; return(MDC_NO); }

  /* initialize new structs */
  for (i=begin; i<number; i++) MdcInitAD(&fi->acqdata[i]);

  /* set new number */
  fi->acqnr = number;

  return(MDC_YES);

}

/* returns 1 on success and 0 on failure        */
int MdcGetStructDD(FILEINFO *fi, Uint32 number)
{
  Uint32 i, begin=number;

  if (number == 0) return(MDC_NO);

  if (fi->dyndata == NULL) {
    /* fresh allocation */
    fi->dyndata = (DYNAMIC_DATA *)malloc(sizeof(DYNAMIC_DATA)*number);
    begin = 0;
  }else if (number != fi->dynnr) {
    /* reallocation */
    fi->dyndata = 
           (DYNAMIC_DATA *)realloc(fi->dyndata,sizeof(DYNAMIC_DATA)*number);
    begin = number > fi->dynnr ? fi->dynnr : number;
  }

  if (fi->dyndata == NULL) { fi->dynnr=0; return(MDC_NO); }

  /* initialize new structs */
  for (i=begin; i<number; i++) MdcInitDD(&fi->dyndata[i]);

  /* set new number */
  fi->dynnr = number;

  return(MDC_YES);
}

/* returns 1 on success and 0 on failure        */
int MdcGetStructBD(FILEINFO *fi, Uint32 number)
{
  Uint32 i, begin=number;

  if (number == 0) return(MDC_NO);

  if (fi->beddata == NULL) {
    /* fresh allocation */
    fi->beddata = (BED_DATA *)malloc(sizeof(BED_DATA)*number);
    begin = 0;
  }else if (number != fi->bednr) {
    /* reallocation */
    fi->beddata = 
           (BED_DATA *)realloc(fi->beddata,sizeof(BED_DATA)*number);
    begin = number > fi->bednr ? fi->bednr : number;
  }

  if (fi->beddata == NULL) { fi->bednr=0; return(MDC_NO); }

  /* initialize new structs */
  for (i=begin; i<number; i++) MdcInitBD(&fi->beddata[i]);

  /* set new number */
  fi->bednr = number;

  return(MDC_YES);
}

void MdcInitMOD(MOD_INFO *mod)
{
  if (mod == NULL) return;

  mod->gn_info.study_date[0]='\0';
  mod->gn_info.study_time[0]='\0';
  mod->gn_info.series_date[0]='\0';
  mod->gn_info.series_time[0]='\0';
  mod->gn_info.acquisition_date[0]='\0';
  mod->gn_info.acquisition_time[0]='\0';
  mod->gn_info.image_date[0]='\0';
  mod->gn_info.image_time[0]='\0';

  mod->mr_info.repetition_time=0.;
  mod->mr_info.echo_time=0.;
  mod->mr_info.inversion_time=0.;
  mod->mr_info.num_averages=0.;
  mod->mr_info.imaging_freq=0.;
  mod->mr_info.pixel_bandwidth=0.;
  mod->mr_info.flip_angle=0.;
  mod->mr_info.dbdt=0.;
  mod->mr_info.transducer_freq=0;
  mod->mr_info.transducer_type[0]='\0';
  mod->mr_info.pulse_repetition_freq=0;
  mod->mr_info.pulse_seq_name[0]='\0';
  mod->mr_info.steady_state_pulse_seq[0]='\0';
  mod->mr_info.slab_thickness=0.;
  mod->mr_info.sampling_freq=0.;

}

void MdcInitID(IMG_DATA *id)
{
  int i;

  if (id == NULL) return;

  memset(id,'\0',sizeof(IMG_DATA));

  id->rescaled = MDC_NO;
  id->quant_scale = 1.; id->calibr_fctr = 1.; id->intercept = 0.;
  id->rescale_slope = 1.; id->rescale_intercept = 0.;
  id->quant_units = 1; id->calibr_units = 1;
  id->frame_number = 0;
  id->slice_start = 0.;
  id->buf = NULL;
  id->load_location = -1;

  id->pixel_xsize = 1.;
  id->pixel_ysize = 1.;
  id->slice_width = 1.;

  for(i=0;i<3;i++) { id->image_pos_dev[i]=0.;    id->image_pos_pat[i]=0.; }
  for(i=0;i<6;i++) { id->image_orient_dev[i]=0.; id->image_orient_pat[i]=0.; }
  id->slice_spacing = 0.; /* = slice_width; when not found no gap */ 
  id->ct_zoom_fctr  = 1.;
 
  id->sdata = NULL;
 
  id->plugb = NULL;
}

void MdcInitSD(STATIC_DATA *sd)
{
  strcpy(sd->label,"Unknown");
  sd->total_counts      = 0.;
  sd->image_duration    = 0.;
  sd->start_time_hour   = 0;
  sd->start_time_minute = 0;
  sd->start_time_second = 0;
}

void MdcInitGD(GATED_DATA *gd) 
{
  if (gd == NULL) return;

  gd->gspect_nesting = MDC_GSPECT_NESTING_GATED;
  gd->nr_projections = 0.0;
  gd->extent_rotation = 0.0;
  gd->study_duration = 0.0;
  gd->image_duration = 0.0;
  gd->time_per_proj  = 0.0;
  gd->window_low = 0.0;
  gd->window_high= 0.0;
  gd->cycles_observed = 0.0;
  gd->cycles_acquired = 0.0;

}

void MdcInitAD(ACQ_DATA *acq)
{
  if (acq == NULL) return;

  acq->rotation_direction = MDC_ROTATION_CW;
  acq->detector_motion = MDC_MOTION_STEP;
  acq->rotation_offset = 0.;
  acq->radial_position = 0.;
  acq->angle_start = 0.;
  acq->angle_step = 0.; 
  acq->scan_arc = 360.;

}

void MdcInitDD(DYNAMIC_DATA *dd)
{
  if (dd == NULL) return;

  dd->nr_of_slices = 0;
  dd->time_frame_start = 0.;
  dd->time_frame_delay = 0.;
  dd->time_frame_duration = 0.;
  dd->delay_slices = 0.;
 
}

void MdcInitBD(BED_DATA *bd)
{
  if (bd == NULL) return;

  bd->hoffset = 0.;
  bd->voffset = 0.;

}

void MdcInitFI(FILEINFO *fi, const char *path)
{  
   fi->ifp = NULL; fi->ifp_raw = NULL; 
   fi->ofp = NULL; fi->ofp_raw = NULL;
   fi->idir = NULL; fi->ifname = NULL;
   fi->odir = NULL; fi->ofname = NULL;
   fi->image = NULL;
   fi->iformat = MDC_FRMT_NONE; fi->oformat = MDC_FRMT_NONE;
   fi->diff_type = MDC_NO; fi->diff_size = MDC_NO; fi->diff_scale = MDC_NO;
   fi->rawconv = MDC_NO;
   fi->endian = MDC_UNKNOWN;
   fi->modality = M_NM;
   fi->compression = MDC_NO;
   fi->truncated=MDC_NO;
   fi->number = 0;
   fi->mwidth=fi->mheight=0;
   fi->bits = 8; fi->type = BIT8_U;
   fi->ifname = fi->ipath;
   memset(fi->ipath,'\0',MDC_MAX_PATH);
   strncpy(fi->ipath,path,MDC_MAX_PATH);
   fi->ofname = fi->opath;
   memset(fi->opath,'\0',MDC_MAX_PATH);
   fi->study_date_day   = 0;
   fi->study_date_month = 0;
   fi->study_date_year  = 0;
   fi->study_time_hour  = 0;
   fi->study_time_minute= 0;
   fi->study_time_second= 0;
   fi->dose_time_hour   = 0;
   fi->dose_time_minute = 0;
   fi->dose_time_second = 0;

   fi->nr_series      = -1;
   fi->nr_acquisition = -1;
   fi->nr_instance    = -1;

   fi->decay_corrected   = MDC_NO;
   fi->flood_corrected   = MDC_NO;
   fi->acquisition_type  = MDC_ACQUISITION_UNKNOWN;
   fi->planar            = MDC_NO;
   fi->reconstructed     = MDC_YES;

   fi->contrast_remapped = MDC_NO;
   fi->window_centre = 0.;
   fi->window_width  = 0.;

   fi->slice_projection = MDC_UNKNOWN;
   fi->pat_slice_orient = MDC_UNKNOWN;

   strcpy(fi->pat_pos,"Unknown");
   strcpy(fi->pat_orient,"Unknown");
   strcpy(fi->recon_method,"Unknown");
   strcpy(fi->patient_name,"Unknown");
   strcpy(fi->patient_id,"Unknown");
   strcpy(fi->patient_sex,"Unknown");
   strcpy(fi->patient_dob,"00000000");
   strcpy(fi->operator_name,"Unknown");
   strcpy(fi->study_descr,"Unknown");
   strcpy(fi->study_id,"Unknown");
   strcpy(fi->institution,MDC_INSTITUTION);
   strcpy(fi->manufacturer,MDC_PRGR);
   strcpy(fi->series_descr,"Unknown");
   strcpy(fi->radiopharma,"Unknown");
   strcpy(fi->filter_type,"Unknown");
   strcpy(fi->organ_code,"Unknown");
   strcpy(fi->isotope_code,"Unknown");
 
   fi->patient_weight   = 0.;
   fi->patient_height   = 0.;
   fi->isotope_halflife = 0.;
   fi->injected_dose    = 0.;
   fi->gantry_tilt      = 0.;

   fi->dim[0] = 3;
   fi->dim[1] = 1;
   fi->dim[2] = 1;
   fi->dim[3] = 1;
   fi->dim[4] = 1;
   fi->dim[5] = 1;
   fi->dim[6] = 1;
   fi->dim[7] = 1;
   fi->pixdim[0] = 3.;
   fi->pixdim[1] = 1.;
   fi->pixdim[2] = 1.;
   fi->pixdim[3] = 1.;
   fi->pixdim[4] = 1.;
   fi->pixdim[5] = 1.;
   fi->pixdim[6] = 1.;
   fi->pixdim[7] = 1.;


   fi->map = MDC_MAP_GRAY;
   MdcGetColorMap((int)fi->map,fi->palette);
   fi->comment = NULL;
   fi->comm_length = 0;
   fi->glmin = fi->glmax = fi->qglmin = fi->qglmax = 0.;

   fi->gatednr = 0; fi->gdata = NULL;

   fi->acqnr = 0; fi->acqdata = NULL;

   fi->dynnr = 0; fi->dyndata = NULL;

   fi->bednr = 0; fi->beddata = NULL;

   fi->mod = NULL;

   fi->pluga = NULL;

}

char *MdcCopyMOD(MOD_INFO *dest, MOD_INFO *src)
{
  GN_INFO *dgn, *sgn;
  MR_INFO *dmr, *smr;

  dgn = &dest->gn_info; sgn = &src->gn_info;

  strncpy(dgn->study_date      , sgn->study_date         , MDC_MAXSTR);
  strncpy(dgn->study_time      , sgn->study_time         , MDC_MAXSTR);
  strncpy(dgn->series_date     , sgn->series_date        , MDC_MAXSTR);
  strncpy(dgn->series_time     , sgn->series_time        , MDC_MAXSTR);
  strncpy(dgn->acquisition_date, sgn->acquisition_date   , MDC_MAXSTR);
  strncpy(dgn->acquisition_time, sgn->acquisition_time   , MDC_MAXSTR);
  strncpy(dgn->image_date      , sgn->image_date         , MDC_MAXSTR);
  strncpy(dgn->image_time      , sgn->image_time         , MDC_MAXSTR);

  dmr = &dest->mr_info; smr = &src->mr_info;

  dmr->repetition_time = smr->repetition_time;
  dmr->echo_time       = smr->echo_time;
  dmr->inversion_time  = smr->inversion_time;
  dmr->num_averages    = smr->num_averages;
  dmr->imaging_freq    = smr->imaging_freq;
  dmr->pixel_bandwidth = smr->pixel_bandwidth;
  dmr->flip_angle      = smr->flip_angle;
  dmr->dbdt            = smr->dbdt;
  dmr->transducer_freq = smr->transducer_freq;
  strncpy(dmr->transducer_type,smr->transducer_type, MDC_MAXSTR);
  dmr->pulse_repetition_freq = smr->pulse_repetition_freq;
  strncpy(dmr->pulse_seq_name, smr->pulse_seq_name,MDC_MAXSTR);
  strncpy(dmr->steady_state_pulse_seq,smr->steady_state_pulse_seq, MDC_MAXSTR);
  dmr->slab_thickness  = smr->slab_thickness;
  dmr->sampling_freq   = smr->sampling_freq;

  return(NULL);

}

char *MdcCopySD(STATIC_DATA *dest, STATIC_DATA *src)
{
  strncpy(dest->label,src->label,MDC_MAXSTR);
  dest->total_counts   = src->total_counts;
  dest->image_duration = src->image_duration;
  dest->start_time_hour   = src->start_time_hour;
  dest->start_time_minute = src->start_time_minute;
  dest->start_time_second = src->start_time_second;

  return(NULL);

}

char *MdcCopyGD(GATED_DATA *dest, GATED_DATA *src)
{
  dest->gspect_nesting  = src->gspect_nesting;
  dest->nr_projections  = src->nr_projections;
  dest->extent_rotation = src->extent_rotation;
  dest->study_duration  = src->study_duration;
  dest->image_duration  = src->image_duration;
  dest->time_per_proj   = src->time_per_proj;
  dest->window_low      = src->window_low;
  dest->window_high     = src->window_high;
  dest->cycles_observed = src->cycles_observed;
  dest->cycles_acquired = src->cycles_acquired;

  return(NULL);
}

char *MdcCopyAD(ACQ_DATA *dest, ACQ_DATA *src)
{
  dest->rotation_direction = src->rotation_direction;
  dest->detector_motion    = src->detector_motion;
  dest->rotation_offset    = src->rotation_offset;
  dest->radial_position    = src->radial_position;
  dest->angle_start        = src->angle_start;
  dest->angle_step         = src->angle_step;
  dest->scan_arc           = src->scan_arc;

  return(NULL);
}

char *MdcCopyDD(DYNAMIC_DATA *dest, DYNAMIC_DATA *src)
{
  dest->nr_of_slices        = src->nr_of_slices;
  dest->time_frame_start    = src->time_frame_start;
  dest->time_frame_delay    = src->time_frame_delay;
  dest->time_frame_duration = src->time_frame_duration;
  dest->delay_slices        = src->delay_slices;

  return(NULL);
}

char *MdcCopyBD(BED_DATA *dest, BED_DATA *src)
{
  dest->hoffset             = src->hoffset;
  dest->voffset             = src->voffset;

  return(NULL);
}

char *MdcCopyID(IMG_DATA *dest, IMG_DATA *src, int COPY_IMAGE)
{
  Uint32 i, w, h, b, size;

  dest->width = src->width;    dest->height = src->height;
  dest->bits  = src->bits;     dest->type   = src->type;
  dest->flags = src->flags;
  dest->min   = src->min;      dest->max    = src->max;
  dest->qmin  = src->qmin;     dest->qmax   = src->qmax;
  dest->fmin  = src->fmin;     dest->fmax   = src->fmax;
  dest->qfmin = src->qfmin;    dest->qfmax  = src->qfmax;

  if (COPY_IMAGE == MDC_YES) {

    dest->rescale_slope = src->rescale_slope;
    dest->rescale_intercept = src->rescale_intercept;

    w = dest->width; h = dest->height; b = MdcType2Bytes(dest->type);
    size = w * h * b;
    dest->buf = malloc(size);
    if (dest->buf == NULL)  return("Failed to copy image buffer");
    memcpy(dest->buf,src->buf,size);
    dest->load_location = src->load_location;
    dest->rescaled      = src->rescaled;
    dest->rescaled_min  = src->rescaled_min;
    dest->rescaled_max  = src->rescaled_max;
    dest->rescaled_fctr = src->rescaled_fctr;
    dest->rescaled_slope= src->rescaled_slope;
    dest->rescaled_intercept = src->rescaled_intercept;

    dest->quant_scale   = src->quant_scale;
    dest->calibr_fctr   = src->calibr_fctr;
    dest->intercept     = src->intercept;

  }else{
    dest->rescale_slope = 1.;
    dest->rescale_intercept = 0.;
    dest->buf = NULL;
    dest->load_location = -1;
    dest->rescaled = MDC_NO;
    dest->rescaled_min  = 0.;
    dest->rescaled_max  = 0.;
    dest->rescaled_fctr = 1.;
    dest->rescaled_slope= 1.;
    dest->rescaled_intercept = 0.;

    dest->quant_scale = 1.;
    dest->calibr_fctr = 1.;
    dest->intercept = 0.;
  }

  dest->frame_number = src->frame_number;
  dest->slice_start  = src->slice_start;

  dest->quant_units  = src->quant_units;
  dest->calibr_units = src->calibr_units;

  dest->pixel_xsize  = src->pixel_xsize;
  dest->pixel_ysize  = src->pixel_ysize;
  dest->slice_width  = src->slice_width;
  dest->recon_scale  = src->recon_scale;

  for (i=0; i<3; i++) dest->image_pos_dev[i] = src->image_pos_dev[i];
  for (i=0; i<6; i++) dest->image_orient_dev[i] = src->image_orient_dev[i];
  for (i=0; i<3; i++) dest->image_pos_pat[i] = src->image_pos_pat[i];
  for (i=0; i<6; i++) dest->image_orient_pat[i] = src->image_orient_pat[i];

  dest->slice_spacing = src->slice_spacing;
  dest->ct_zoom_fctr  = src->ct_zoom_fctr;

  /* static data */
  if (src->sdata != NULL) {
    dest->sdata = (STATIC_DATA *)malloc(sizeof(STATIC_DATA));
    if (dest->sdata == NULL) return("Failed to copy static data struct");
    MdcCopySD(dest->sdata,src->sdata);
  }else{
    dest->sdata = NULL;
  }

  /* no copying here; just initialize plugb */
  dest->plugb = NULL;

  return(NULL);
}

/* KEEP_FILES = preserve file pointers; src pointers are masked (!) */
char *MdcCopyFI(FILEINFO *dest, FILEINFO *src, int COPY_IMAGES, int KEEP_FILES)
{
  char *msg=NULL;
  int i;

  MdcInitFI(dest,src->ifname);

  if (KEEP_FILES == MDC_YES) {
    /* copy pointers */
    dest->ifp = src->ifp;
    dest->ifp_raw = src->ifp_raw;
    dest->ofp = src->ofp;
    dest->ofp_raw = src->ofp_raw;
    /* mask src pointers */
    src->ifp = NULL; src->ifp_raw = NULL;
    src->ofp = NULL; src->ofp_raw = NULL;
  }

  /* A) src reassemble */
  MdcMergePath(src->ipath,src->idir,src->ifname);
  MdcMergePath(src->opath,src->odir,src->ofname);

  /* B) src -> dest */
  memcpy(dest->ipath,src->ipath,MDC_MAX_PATH);
  memcpy(dest->opath,src->opath,MDC_MAX_PATH);

  /* C) dest disassemble */
  MdcSplitPath(dest->ipath,dest->idir,dest->ifname);
  MdcSplitPath(dest->opath,dest->odir,dest->ofname);

  /* D) src disassemble, undo A) */
  MdcSplitPath(src->ipath,src->idir,src->ifname);
  MdcSplitPath(src->opath,src->odir,src->ofname);

  dest->iformat = src->iformat;
  dest->oformat = src->oformat;
  dest->rawconv = src->rawconv;
  dest->endian  = src->endian;
  dest->modality    = src->modality;
  dest->compression = src->compression;
  dest->truncated   = src->truncated;
  dest->diff_type   = src->diff_type;
  dest->diff_size   = src->diff_size;
  dest->diff_scale  = src->diff_scale;
/*dest->number      = src->number;*/     /* just see later  */
  dest->mwidth      = src->mwidth;
  dest->mheight     = src->mheight;
  dest->bits        = src->bits;
  dest->type        = src->type;

  for (i=0; i<MDC_MAX_DIMS; i++) dest->dim[i]    = src->dim[i];
  for (i=0; i<MDC_MAX_DIMS; i++) dest->pixdim[i] = src->pixdim[i]; 

  dest->glmin       = src->glmin;
  dest->glmax       = src->glmax;
  dest->qglmin      = src->qglmin;
  dest->qglmax      = src->qglmax;

  dest->contrast_remapped = src->contrast_remapped;
  dest->window_centre = src->window_centre;
  dest->window_width  = src->window_width;

  dest->slice_projection = src->slice_projection;
  dest->pat_slice_orient = src->pat_slice_orient;

  strncpy(dest->pat_pos,src->pat_pos,MDC_MAXSTR);
  strncpy(dest->pat_orient,src->pat_orient,MDC_MAXSTR);
  strncpy(dest->patient_sex,src->patient_sex,MDC_MAXSTR);
  strncpy(dest->patient_name,src->patient_name,MDC_MAXSTR);
  strncpy(dest->patient_id,src->patient_id,MDC_MAXSTR);
  strncpy(dest->patient_dob,src->patient_dob,MDC_MAXSTR);
  strncpy(dest->operator_name,src->operator_name,MDC_MAXSTR);
  strncpy(dest->study_descr,src->study_descr,MDC_MAXSTR);
  strncpy(dest->study_id,src->study_id,MDC_MAXSTR);

  dest->study_date_day   = src->study_date_day;
  dest->study_date_month = src->study_date_month;
  dest->study_date_year  = src->study_date_year;
  dest->study_time_hour  = src->study_time_hour;
  dest->study_time_minute= src->study_time_minute;
  dest->study_time_second= src->study_time_second;
  dest->dose_time_hour   = src->dose_time_hour;
  dest->dose_time_minute = src->dose_time_minute;
  dest->dose_time_second = src->dose_time_second;
  dest->nr_series        = src->nr_series;
  dest->nr_acquisition   = src->nr_acquisition;
  dest->nr_instance      = src->nr_instance;
  dest->acquisition_type = src->acquisition_type;
  dest->planar           = src->planar;
  dest->decay_corrected  = src->decay_corrected;
  dest->flood_corrected  = src->flood_corrected;
  dest->reconstructed    = src->reconstructed;

  strncpy(dest->recon_method,src->recon_method,MDC_MAXSTR);
  strncpy(dest->institution,src->institution,MDC_MAXSTR);
  strncpy(dest->manufacturer,src->manufacturer,MDC_MAXSTR);
  strncpy(dest->series_descr,src->series_descr,MDC_MAXSTR);
  strncpy(dest->radiopharma,src->radiopharma,MDC_MAXSTR);
  strncpy(dest->filter_type,src->filter_type,MDC_MAXSTR);
  strncpy(dest->organ_code,src->organ_code,MDC_MAXSTR);
  strncpy(dest->isotope_code,src->isotope_code,MDC_MAXSTR);

  dest->patient_weight   = src->patient_weight;
  dest->patient_height   = src->patient_height;
  dest->isotope_halflife = src->isotope_halflife;
  dest->gantry_tilt      = src->gantry_tilt;
  dest->injected_dose    = src->injected_dose;

  dest->map = src->map;
  memcpy(dest->palette,src->palette,768);

  /* copy comment */
  if (src->comm_length > 0) {
    dest->comment = malloc(src->comm_length);
    if (dest->comment == NULL) {
      /* bad, but don't fail on some comment */
      dest->comm_length = 0;
    }else{
      dest->comm_length = src->comm_length;
      memcpy(dest->comment,src->comment,dest->comm_length);
    }
  }else{
   dest->comm_length = 0; dest->comment = NULL; 
  }

  /* copy ACQ_DATA structs */
  if (src->acqnr > 0 && src->acqdata != NULL) {
    dest->acqnr = src->acqnr;
    dest->acqdata = (ACQ_DATA *)malloc(dest->acqnr * sizeof(ACQ_DATA));
    if (dest->acqdata == NULL) return("Failed to create ACQ_DATA structs");
    for (i=0; i<dest->acqnr; i++) {
       msg = MdcCopyAD(&dest->acqdata[i],&src->acqdata[i]);
       if (msg != NULL) return(msg);
    }
  }else{
    dest->acqnr = 0; dest->acqdata = NULL;
  }

  /* copy GATED_DATA structs */
  if (src->gatednr > 0 && src->gdata != NULL) {
    dest->gatednr = src->gatednr;
    dest->gdata = (GATED_DATA *)malloc(dest->gatednr * sizeof(GATED_DATA));
    if (dest->gdata == NULL) return("Failed to create GATED_DATA structs");
    for (i=0; i<dest->gatednr; i++) {
       msg = MdcCopyGD(&dest->gdata[i],&src->gdata[i]);
       if (msg != NULL) return(msg);
    }
  }else{
    dest->gatednr = 0; dest->gdata = NULL;
  }

  /* copy DYNAMIC_DATA structs */
  if ((src->dynnr > 0) && (src->dyndata != NULL)) {
    dest->dynnr = src->dynnr;
    dest->dyndata = (DYNAMIC_DATA *)malloc(dest->dynnr * sizeof(DYNAMIC_DATA));
    if (dest->dyndata == NULL) return("Failed to create DYNAMIC_DATA structs");
    for (i=0; i<dest->dynnr; i++) {
       msg = MdcCopyDD(&dest->dyndata[i],&src->dyndata[i]);
       if (msg != NULL) return(msg);
    }
  }else{
    dest->dynnr = 0; dest->dyndata = NULL;
  }

  /* copy BED_DATA structs */
  if ((src->bednr > 0) && (src->beddata != NULL)) {
    dest->bednr = src->bednr;
    dest->beddata = (BED_DATA *)malloc(dest->bednr * sizeof(BED_DATA));
    if (dest->beddata == NULL) return("Failed to create BED_DATA structs");
    for (i=0; i<dest->bednr; i++) {
       msg = MdcCopyBD(&dest->beddata[i],&src->beddata[i]);
       if (msg != NULL) return(msg);
    }
  }else{
    dest->bednr = 0; dest->beddata = NULL;
  }

  /* copy IMG_DATA structs */
  if ((COPY_IMAGES == MDC_YES) && (src->number > 0) && (src->image != NULL)) {
    dest->number = src->number;
    dest->image = (IMG_DATA *)malloc(dest->number * sizeof(IMG_DATA));
    if (dest->image == NULL) return("Failed to create IMG_DATA structs");
    for (i=0; i<dest->number; i++) {
       msg = MdcCopyID(&dest->image[i],&src->image[i],MDC_YES);
       if (msg != NULL) return(msg);
    } 
  }else{
    dest->number = 0; dest->image = NULL;
  }
 
  /* copy MOD_INFO struct */
  if (src->mod != NULL) {
    dest->mod = (MOD_INFO *)malloc(sizeof(MOD_INFO));
    if (dest->mod == NULL) return("Failed to copy MOD_INFO struct");
    MdcCopyMOD(dest->mod,src->mod);
  }else{
    dest->mod = NULL;
  }

  return(NULL);
}

void MdcFreeMODs(FILEINFO *fi)
{
  MdcFree(fi->mod);
}

void MdcFreeIDs(FILEINFO *fi)
{ 
  IMG_DATA *id=NULL;
  Uint32 i;
 
  if ( fi->image != NULL ) {
    for ( i=0; i<fi->number; i++) {
       id = (IMG_DATA *)&fi->image[i];
       MdcFree(id->buf);
       MdcFree(id->sdata);
       MdcFree(id->plugb);
    }
    MdcFree(fi->image);
  }
}

void MdcFreeODs(FILEINFO *fi)
{
  Uint32 i;

  if (fi->acqnr > 0)   { MdcFree(fi->acqdata);   fi->acqnr = 0;   }

  if (fi->dynnr > 0)   { MdcFree(fi->dyndata);   fi->dynnr = 0;   }

  if (fi->bednr > 0)   { MdcFree(fi->beddata);   fi->bednr = 0;   }

  if (fi->gatednr > 0) { MdcFree(fi->gdata);     fi->gatednr = 0; }

  for (i=0; i<fi->number; i++) MdcFree(fi->image[i].sdata);
}
 
void MdcResetIDs(FILEINFO *fi)
{
  Uint32 i;

  for (i=0; i<fi->number; i++)  {
     fi->image[i].rescaled = MDC_NO;
     fi->image[i].rescaled_max  = 0.;
     fi->image[i].rescaled_min  = 0.;
     fi->image[i].rescaled_fctr = 1.;
     fi->image[i].rescaled_slope= 1.;
     fi->image[i].rescaled_intercept = 0.;
  }

}

char *MdcResetODs(FILEINFO *fi)
{
  Uint32 i;

  /* first free other data structs */
  MdcFreeODs(fi);

  /* now get emtpy structs */ 
  if (fi->reconstructed == MDC_NO) {
    if (!MdcGetStructAD(fi,1))
      return("Failure to reset ACQ_DATA structs");
  }

  if ((fi->acquisition_type == MDC_ACQUISITION_GATED ||
       fi->acquisition_type == MDC_ACQUISITION_GSPECT) && (fi->gatednr == 0)) {
    if (!MdcGetStructGD(fi,1))
      return("Failure to reset GATED_DATA structs");
  }

  if ((fi->acquisition_type == MDC_ACQUISITION_DYNAMIC ||
       fi->acquisition_type == MDC_ACQUISITION_TOMO) && (fi->dynnr == 0)) {
    if (!MdcGetStructDD(fi,(Uint32)fi->dim[4]))
      return("Failure to reset DYNAMIC_DATA structs");

    for (i=0; i<fi->dynnr; i++) {
       fi->dyndata[i].nr_of_slices = fi->dim[3];
       fi->dyndata[i].time_frame_duration = fi->pixdim[4];
    }
  }

  if (fi->bednr == 0) {
    if (!MdcGetStructBD(fi,(Uint32)fi->dim[6]))
      return("Failure to reset BED_DATA structs");

    for (i=0; i<fi->bednr; i++) {
       fi->beddata[i].hoffset = 0.;
       fi->beddata[i].voffset = 0.;
    }
  }

  if ((fi->acquisition_type == MDC_ACQUISITION_STATIC) && (fi->number > 0)) {
    if (!MdcGetStructSD(fi,fi->number))
      return("Failure to reset STATIC_DATA structs");
  }

  return(NULL);

}

void MdcCleanUpFI(FILEINFO  *fi)
{
  if (fi->dynnr   > 0) { MdcFree(fi->dyndata);     fi->dynnr = 0; }
  if (fi->acqnr   > 0) { MdcFree(fi->acqdata);     fi->acqnr = 0; }
  if (fi->bednr   > 0) { MdcFree(fi->beddata);     fi->bednr = 0; }
  if (fi->gatednr > 0) { MdcFree(fi->gdata);       fi->gatednr = 0; }
  if (fi->comm_length > 0) { MdcFree(fi->comment); fi->comm_length = 0; }

  MdcFreeIDs(fi);

  MdcFreeMODs(fi);

  MdcFree(fi->pluga);

  MdcCloseFile(fi->ifp);
  MdcCloseFile(fi->ifp_raw);
  MdcCloseFile(fi->ofp);
  MdcCloseFile(fi->ofp_raw);

  MdcInitFI(fi,"<null>");

}

