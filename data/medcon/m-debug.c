/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * filename: m-debug.c                                                     *
 *                                                                         * 
 * UTIL C-source: Medical Image Conversion Utility                         *
 *                                                                         *
 * purpose      : print FILEINFO structure                                 *
 *                                                                         *
 * project      : (X)MedCon by Erik Nolf                                   *
 *                                                                         *
 * Functions    : MdcPrintFI()       - Display FILEINFO struct             *
 *                MdcDebugPrint()    - Print MDC_MY_DEBUG info             *
 *                                                                         *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* $Id: m-debug.c,v 1.69 2010/08/28 23:44:23 enlf Exp $
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

#include <stdio.h>
#include <stdarg.h>

#include "m-defs.h"
#include "m-fancy.h"
#include "m-files.h"
#include "m-debug.h"

#if GLIBSUPPORTED
#include <glib.h>
#endif

/****************************************************************************
                            F U N C T I O N S
****************************************************************************/

void MdcPrintFI(FILEINFO *fi)
{ 
  Uint32 i, j;
  int v;
  float f;
  IMG_DATA *id;

  MdcPrntScrn("\n");
  MdcPrintLine('#',MDC_FULL_LENGTH);
  MdcPrntScrn("FILEINFO - Global Data\n");
  MdcPrintLine('#',MDC_FULL_LENGTH);
  MdcPrntScrn("FILE *ifp          : ");
  if (fi->ifp == NULL) MdcPrntScrn("<null>\n");
  else MdcPrntScrn("%p\n",fi->ifp);
  MdcPrntScrn("FILE *ofp          : ");
  if (fi->ofp == NULL) MdcPrntScrn("<null>\n");
  else MdcPrntScrn("%p\n",fi->ofp);
  MdcPrntScrn("ipath              : %s\n",fi->ipath);
  MdcPrntScrn("opath              : %s\n",fi->opath);
  if (fi->idir == NULL) 
  MdcPrntScrn("idir               : <null>\n");
  else
  MdcPrntScrn("idir               : %s\n",fi->idir);
  if (fi->odir == NULL)
  MdcPrntScrn("odir               : <null>\n");
  else
  MdcPrntScrn("odir               : %s\n",fi->odir);
  MdcPrntScrn("ifname             : %s\n",fi->ifname);
  MdcPrntScrn("ofname             : %s\n",fi->ofname);
  MdcPrntScrn("iformat            : %d (= %s)\n",fi->iformat
                                                ,FrmtString[fi->iformat]);
  MdcPrntScrn("oformat            : %d (= %s)\n",fi->oformat
                                                ,FrmtString[fi->oformat]);
  MdcPrntScrn("modality           : %d (= %s)\n",fi->modality
                                            ,MdcGetStrModality(fi->modality));
  v = (int)fi->rawconv;
  MdcPrntScrn("rawconv            : %d (= %s)\n",v,MdcGetStrRawConv(v));
  v = (int)fi->endian;
  MdcPrntScrn("endian             : %d (= %s)\n",v,MdcGetStrEndian(v));
  v = (int)fi->compression;
  MdcPrntScrn("compression        : %d (= %s)\n",v,MdcGetStrCompression(v));
  MdcPrntScrn("truncated          : %d ",fi->truncated);
  MdcPrintYesNo(fi->truncated);
  MdcPrntScrn("diff_type          : %d ",fi->diff_type);
  MdcPrintYesNo(fi->diff_type);
  MdcPrntScrn("diff_size          : %d ",fi->diff_size);
  MdcPrintYesNo(fi->diff_size);
  MdcPrntScrn("diff_scale         : %d ",fi->diff_scale);
  MdcPrintYesNo(fi->diff_scale);
  MdcPrntScrn("number             : %u\n",fi->number);
  MdcPrntScrn("mwidth             : %u\n",fi->mwidth);
  MdcPrntScrn("mheight            : %u\n",fi->mheight);
  MdcPrntScrn("bits               : %hu\n",fi->bits);
  v = (int)fi->type;
  MdcPrntScrn("type               : %d (= %s)\n",v,MdcGetStrPixelType(v));
  MdcPrntScrn("dim[0]             : %-5hd (= total in use)\n",fi->dim[0]);
  MdcPrntScrn("dim[1]             : %-5hd (= pixels X-dim)\n",fi->dim[1]);
  MdcPrntScrn("dim[2]             : %-5hd (= pixels Y-dim)\n",fi->dim[2]);
  MdcPrntScrn("dim[3]             : %-5hd (= planes | (time) slices)\n"
                                                                ,fi->dim[3]);
  MdcPrntScrn("dim[4]             : %-5hd (= frames | time slots | phases)\n"
                                                                ,fi->dim[4]);
  MdcPrntScrn("dim[5]             : %-5hd (= gates  | R-R intervals)\n"
                                                                ,fi->dim[5]);
  MdcPrntScrn("dim[6]             : %-5hd (= beds   | detector heads)\n"
                                                                ,fi->dim[6]);
  MdcPrntScrn("dim[7]             : %-5hd (= ...    | energy windows)\n"
                                                                ,fi->dim[7]);
  MdcPrntScrn("pixdim[0]          : %+e\n",fi->pixdim[0]);
  MdcPrntScrn("pixdim[1]          : %+e [mm]\n",fi->pixdim[1]);
  MdcPrntScrn("pixdim[2]          : %+e [mm]\n",fi->pixdim[2]);
  MdcPrntScrn("pixdim[3]          : %+e [mm]\n",fi->pixdim[3]);
  for (i=4; i<MDC_MAX_DIMS; i++)
  MdcPrntScrn("pixdim[%u]          : %+e\n",i,fi->pixdim[i]);
  MdcPrntScrn("glmin              : %+e\n",fi->glmin);
  MdcPrntScrn("glmax              : %+e\n",fi->glmax);
  MdcPrntScrn("qglmin             : %+e\n",fi->qglmin);
  MdcPrntScrn("qglmax             : %+e\n",fi->qglmax);
  MdcPrntScrn("contrast_remapped  : %hd ",fi->contrast_remapped);
  MdcPrintYesNo(fi->contrast_remapped);
  MdcPrntScrn("window_centre      : %g\n",fi->window_centre);
  MdcPrntScrn("window_width       : %g\n",fi->window_width);
  MdcPrntScrn("slice_projection   : %d (= %s)\n",fi->slice_projection,
                               MdcGetStrSlProjection(fi->slice_projection));
  MdcPrntScrn("pat_slice_orient   : %d (= %s)\n",fi->pat_slice_orient,
                               MdcGetStrPatSlOrient(fi->pat_slice_orient));
  MdcPrntScrn("pat_pos            : %s\n",fi->pat_pos);
  MdcPrntScrn("pat_orient         : %s\n",fi->pat_orient);
  MdcPrntScrn("patient_sex        : %s\n",fi->patient_sex);
  MdcPrntScrn("patient_name       : %s\n",fi->patient_name);
  MdcPrntScrn("patient_id         : %s\n",fi->patient_id);
  MdcPrntScrn("patient_dob        : %s\n",fi->patient_dob);
  MdcPrntScrn("patient_weight     : %.2f [kg]\n",fi->patient_weight);
  MdcPrntScrn("patient_height     : %.2f [m]\n",fi->patient_height);
  MdcPrntScrn("operator_name      : %s\n",fi->operator_name);
  MdcPrntScrn("study_descr        : %s\n",fi->study_descr);
  MdcPrntScrn("study_id           : %s\n",fi->study_id);
  MdcPrntScrn("study_date_year    : %02d\n",fi->study_date_year);
  MdcPrntScrn("study_date_month   : %02d\n",fi->study_date_month);
  MdcPrntScrn("study_date_day     : %02d\n",fi->study_date_day);
  MdcPrntScrn("study_time_hour    : %02d\n",fi->study_time_hour);
  MdcPrntScrn("study_time_minute  : %02d\n",fi->study_time_minute);
  MdcPrntScrn("study_time_second  : %02d\n",fi->study_time_second);
  MdcPrntScrn("dose_time_hour     : %02d\n",fi->dose_time_hour);
  MdcPrntScrn("dose_time_minute   : %02d\n",fi->dose_time_minute);
  MdcPrntScrn("dose_time_second   : %02d\n",fi->dose_time_second);
  MdcPrntScrn("nr_series          : %-10d ",fi->nr_series);
  if (fi->nr_series < 0) MdcPrintYesNo(MDC_NO);
  else MdcPrintYesNo(MDC_YES);
  MdcPrntScrn("nr_acquisition     : %-10d ",fi->nr_acquisition);
  if (fi->nr_acquisition < 0) MdcPrintYesNo(MDC_NO);
  else MdcPrintYesNo(MDC_YES);
  MdcPrntScrn("nr_instance        : %-10d ",fi->nr_instance);
  if (fi->nr_instance < 0) MdcPrintYesNo(MDC_NO);
  else MdcPrintYesNo(MDC_YES);
  v = fi->acquisition_type;
  MdcPrntScrn("acquisition_type   : %d (= %s)\n",v,MdcGetStrAcquisition(v));
  MdcPrntScrn("planar             : %d ",fi->planar);
  MdcPrintYesNo(fi->planar);
  MdcPrntScrn("decay_corrected    : %d ",fi->decay_corrected);
  MdcPrintYesNo(fi->decay_corrected);
  MdcPrntScrn("flood_corrected    : %d ",fi->flood_corrected);
  MdcPrintYesNo(fi->flood_corrected);
  MdcPrntScrn("reconstructed      : %d ",fi->reconstructed);
  MdcPrintYesNo(fi->reconstructed);
  MdcPrntScrn("recon_method       : %s\n",fi->recon_method);
  MdcPrntScrn("institution        : %s\n",fi->institution);
  MdcPrntScrn("manufacturer       : %s\n",fi->manufacturer);
  MdcPrntScrn("series_descr       : %s\n",fi->series_descr);
  MdcPrntScrn("radiopharma        : %s\n",fi->radiopharma);
  MdcPrntScrn("filter_type        : %s\n",fi->filter_type);
  MdcPrntScrn("organ_code         : %s\n",fi->organ_code);
  MdcPrntScrn("isotope_code       : %s\n",fi->isotope_code);
  MdcPrntScrn("isotope_halflife   : %+e [sec] or %g [hrs]\n"
                                         ,fi->isotope_halflife
                                         ,fi->isotope_halflife/3600.);
  MdcPrntScrn("injected_dose      : %+e [MBq]\n",fi->injected_dose);
  MdcPrntScrn("gantry_tilt        : %+e [degrees]\n",fi->gantry_tilt);
  v = (int) fi->map;
  MdcPrntScrn("map                : %u (= %s)\n",v,MdcGetStrColorMap(v));
  MdcPrntScrn("comm_length        : %u\n",fi->comm_length);
  MdcPrntScrn("comment            : ");
  if ((fi->comment != NULL) && (fi->comm_length != 0)) {
    for (i=0; i<fi->comm_length; i++) MdcPrntScrn("%c",fi->comment[i]);
  }else{
    MdcPrntScrn("<null>");
  }
  MdcPrntScrn("\n");

  /* GATED DATA */
  MdcPrntScrn("\ngatednr            : %u\n",fi->gatednr);
  if (fi->gdata != NULL) {
    for (i=0; i < fi->gatednr; i++) {
       GATED_DATA *gd = &fi->gdata[i];

       MdcPrntScrn("\n");
       MdcPrintLine('-',MDC_FULL_LENGTH);
       MdcPrntScrn("FILEINFO - Gated (SPECT) Data #%.3u\n",i+1);
       MdcPrintLine('-',MDC_FULL_LENGTH);
       MdcPrntScrn("gspect_nesting     : %d (= %s)\n",gd->gspect_nesting
              ,MdcGetStrGSpectNesting(gd->gspect_nesting));
       MdcPrntScrn("nr_projections     : %g\n",gd->nr_projections);
       MdcPrntScrn("extent_rotation    : %g\n",gd->extent_rotation);
       MdcPrntScrn("study_duration     : %+e [ms] = %s\n"
                  ,gd->study_duration,MdcGetStrHHMMSS(gd->study_duration));
       MdcPrntScrn("image_duration     : %+e [ms] = %s\n"
                  ,gd->image_duration,MdcGetStrHHMMSS(gd->image_duration));
       MdcPrntScrn("time_per_proj      : %+e [ms] = %s\n"
                  ,gd->time_per_proj,MdcGetStrHHMMSS(gd->time_per_proj));
       MdcPrntScrn("window_low         : %+e [ms] = %s\n"
                  ,gd->window_low,MdcGetStrHHMMSS(gd->window_low));
       MdcPrntScrn("window_high        : %+e [ms] = %s\n"
                  ,gd->window_high,MdcGetStrHHMMSS(gd->window_high));
       MdcPrntScrn("cycles_observed    : %+e\n",gd->cycles_observed);
       MdcPrntScrn("cycles_acquired    : %+e\n\n",gd->cycles_acquired);
       MdcPrntScrn("heart rate (observed): %d [bpm] (auto-filled)\n"
                        ,(int)MdcGetHeartRate(gd,MDC_HEART_RATE_OBSERVED));
       MdcPrntScrn("heart rate (acquired): %d [bpm] (auto-filled)\n"
                        ,(int)MdcGetHeartRate(gd,MDC_HEART_RATE_ACQUIRED));
    }
  }else{
    MdcPrntScrn("gdata              : <null>\n");
  }
  /* ACQUISITION DATA */
  MdcPrntScrn("\nacqnr              : %u\n",fi->acqnr);
  if (fi->acqdata != NULL) {
    for (i=0; i < fi->acqnr; i++) {
       ACQ_DATA *acq = &fi->acqdata[i];

       MdcPrntScrn("\n");
       MdcPrintLine('-',MDC_FULL_LENGTH);
       MdcPrntScrn("FILEINFO - Acquisition Data #%.3u\n",i+1);
       MdcPrintLine('-',MDC_FULL_LENGTH);
       v = acq->rotation_direction;
       MdcPrntScrn("rotation_direction : %d (= %s)\n",v,MdcGetStrRotation(v));
       v = acq->detector_motion;
       MdcPrntScrn("detector_motion    : %d (= %s)\n",v,MdcGetStrMotion(v));
       MdcPrntScrn("rotation_offset    : %g [mm]\n",acq->rotation_offset);
       MdcPrntScrn("radial_position    : %g [mm]\n",acq->radial_position);
       MdcPrntScrn("angle_start        : %g [degrees]\n",acq->angle_start);
       MdcPrntScrn("angle_step         : %g [degrees]\n",acq->angle_step);
       MdcPrntScrn("scan_arc           : %g [degrees]\n",acq->scan_arc);
    }
  }else{
    MdcPrntScrn("acqdata            : <null>\n");
  }

  /* DYNAMIC DATA */
  MdcPrntScrn("\ndynnr              : %u\n",fi->dynnr);
  if (fi->dyndata != NULL) {
    for (i=0; i < fi->dynnr; i++) {
       DYNAMIC_DATA *dd = &fi->dyndata[i];

       MdcPrntScrn("\n");
       MdcPrintLine('-',MDC_FULL_LENGTH);
       MdcPrntScrn("FILEINFO - Dynamic Data #%.3u\n",i+1);
       MdcPrintLine('-',MDC_FULL_LENGTH);
       MdcPrntScrn("number of slices   : %u\n",dd->nr_of_slices);
       MdcPrntScrn("time_frame_start   : %+e [ms] = %s\n"
             ,dd->time_frame_start,MdcGetStrHHMMSS(dd->time_frame_start));
       MdcPrntScrn("time_frame_delay   : %+e [ms] = %s\n"
             ,dd->time_frame_delay,MdcGetStrHHMMSS(dd->time_frame_delay));
       MdcPrntScrn("time_frame_duration: %+e [ms] = %s\n"
             ,dd->time_frame_duration,MdcGetStrHHMMSS(dd->time_frame_duration));
       MdcPrntScrn("delay_slices       : %+e [ms] = %s\n"
             ,dd->delay_slices,MdcGetStrHHMMSS(dd->delay_slices));
    }
  }else{
    MdcPrntScrn("dyndata            : <null>\n");
  }

  /* BED DATA */
  MdcPrntScrn("\nbednr              : %u\n",fi->bednr);
  if (fi->beddata != NULL) {
    for (i=0; i < fi->bednr; i++) {
       BED_DATA *bd = &fi->beddata[i];

       MdcPrntScrn("\n");
       MdcPrintLine('-',MDC_FULL_LENGTH);
       MdcPrntScrn("FILEINFO - Bed Data #%.3u\n",i+1);
       MdcPrintLine('-',MDC_FULL_LENGTH);
       MdcPrntScrn("hoffset            : %+e [mm]\n"
             ,bd->hoffset);
       MdcPrntScrn("voffset            : %+e [mm]\n"
             ,bd->voffset);
    }
  }else{
    MdcPrntScrn("beddata            : <null>\n");
  }

  /* IMAGE DATA */
  for (i=0; i<fi->number; i++) {
     id = &fi->image[i];
     MdcPrntScrn("\n");
     MdcPrintLine('-',MDC_FULL_LENGTH);
     MdcPrntScrn("FILEINFO - Image Data #%.3u\n",i+1);  
     MdcPrintLine('-',MDC_FULL_LENGTH);
     MdcPrntScrn("width              : %u\n",id->width);
     MdcPrntScrn("height             : %u\n",id->height);
     MdcPrntScrn("bits               : %hd\n",id->bits);
     MdcPrntScrn("type               : %hd (= %s)\n",id->type
                                             ,MdcGetStrPixelType(id->type));
     MdcPrntScrn("flags              : 0x%x\n",id->flags);
     MdcPrntScrn("min                : %+e\n",id->min);
     MdcPrntScrn("max                : %+e\n",id->max);
     MdcPrntScrn("qmin               : %+e\n",id->qmin);
     MdcPrntScrn("qmax               : %+e\n",id->qmax);
     MdcPrntScrn("fmin               : %+e\n",id->fmin);
     MdcPrntScrn("fmax               : %+e\n",id->fmax);
     MdcPrntScrn("qfmin              : %+e\n",id->qfmin);
     MdcPrntScrn("qfmax              : %+e\n",id->qfmax);
     MdcPrntScrn("rescale_slope      : %+e\n",id->rescale_slope);
     MdcPrntScrn("rescale_intercept  : %+e\n",id->rescale_intercept);
     MdcPrntScrn("frame_number       : %u\n",id->frame_number);
     MdcPrntScrn("slice_start        : %+e [ms] = %s\n"
                ,id->slice_start,MdcGetStrHHMMSS(id->slice_start));
     f = MdcSingleImageDuration(fi,id->frame_number-1); 
     MdcPrntScrn("slice_duration     : %+e [ms] = %s (auto-filled)\n"
                ,f,MdcGetStrHHMMSS(f));
     MdcPrntScrn("rescaled           : %d ",id->rescaled);
     MdcPrintYesNo(id->rescaled);
     MdcPrntScrn("rescaled_min       : %+e\n",id->rescaled_min);
     MdcPrntScrn("rescaled_max       : %+e\n",id->rescaled_max);
     MdcPrntScrn("rescaled_fctr      : %+e\n",id->rescaled_fctr);
     MdcPrntScrn("rescaled_slope     : %+e\n",id->rescaled_slope);
     MdcPrntScrn("rescaled_intercept : %+e\n",id->rescaled_intercept);
     MdcPrntScrn("buf                : %p\n",id->buf);
     MdcPrntScrn("load_location      : %ld\n",id->load_location);
     MdcPrntScrn("quant_units        : %hd\n",id->quant_units);
     MdcPrntScrn("calibr_units       : %hd\n",id->calibr_units);
     MdcPrntScrn("quant_scale        : %+e\n",id->quant_scale);
     MdcPrntScrn("calibr_fctr        : %+e\n",id->calibr_fctr);
     MdcPrntScrn("intercept          : %+e\n",id->intercept);
     MdcPrntScrn("pixel_xsize        : %+e [mm]\n",id->pixel_xsize);
     MdcPrntScrn("pixel_ysize        : %+e [mm]\n",id->pixel_ysize);
     MdcPrntScrn("slice_width        : %+e [mm]\n",id->slice_width);
     MdcPrntScrn("recon_scale        : %+e\n",id->recon_scale);
     for (j=0; j<3; j++) MdcPrntScrn("image_pos_dev[%u]   : %+e [mm]\n",j
                                                      ,id->image_pos_dev[j]);
     for (j=0; j<3; j++) MdcPrntScrn("image_pos_pat[%u]   : %+e [mm]\n",j
                                                      ,id->image_pos_pat[j]);
     for (j=0; j<6; j++) MdcPrntScrn("image_orient_dev[%u]: %+e [mm]\n",j
                                                      ,id->image_orient_dev[j]);
     for (j=0; j<6; j++) MdcPrntScrn("image_orient_pat[%u]: %+e [mm]\n",j
                                                      ,id->image_orient_pat[j]);
     MdcPrntScrn("slice_spacing      : %+e [mm]\n",id->slice_spacing);
     MdcPrntScrn("ct_zoom_fctr       : %+e\n",id->ct_zoom_fctr);

     if (id->sdata != NULL) {
       STATIC_DATA *sd = id->sdata;
       MdcPrntScrn("\n");
       MdcPrintLine('-',MDC_HALF_LENGTH);
       MdcPrntScrn("FILEINFO - Static Data #%.3u\n",i+1);
       MdcPrintLine('-',MDC_HALF_LENGTH);
       MdcPrntScrn("label              : %s\n",sd->label);
       MdcPrntScrn("total_counts       : %g\n",sd->total_counts);
       MdcPrntScrn("image_duration     : %+e [ms] = %s\n"
                  ,sd->image_duration,MdcGetStrHHMMSS(sd->image_duration));
       MdcPrntScrn("start_time_hour    : %02hd\n",sd->start_time_hour);
       MdcPrntScrn("start_time_minute  : %02hd\n",sd->start_time_minute);
       MdcPrntScrn("start_time_second  : %02hd\n",sd->start_time_second);
       MdcPrintLine('-',MDC_HALF_LENGTH);
     }
 }

 /* DICOM MOD */
 if (fi->mod != NULL) {
   GN_INFO *gn = &fi->mod->gn_info;
   MR_INFO *mr = &fi->mod->mr_info;

   MdcPrntScrn("\n");
   MdcPrintLine('-',MDC_HALF_LENGTH);
   MdcPrntScrn("FILEINFO - DICOM General Info\n");
   MdcPrintLine('-',MDC_HALF_LENGTH); 

   MdcPrntScrn("study_date            : %s\n",gn->study_date);
   MdcPrntScrn("study_time            : %s\n",gn->study_time);
   MdcPrntScrn("series_date           : %s\n",gn->series_date);
   MdcPrntScrn("series_time           : %s\n",gn->series_time);
   MdcPrntScrn("acquisition_date      : %s\n",gn->acquisition_date);
   MdcPrntScrn("acquisition_time      : %s\n",gn->acquisition_time);
   MdcPrntScrn("image_date            : %s\n",gn->image_date);
   MdcPrntScrn("image_time            : %s\n",gn->image_time);

   switch (fi->modality) {
     case M_MR:
         MdcPrintLine('-',MDC_HALF_LENGTH);
         MdcPrntScrn("FILEINFO - DICOM MR Modality Info\n");
         MdcPrintLine('-',MDC_HALF_LENGTH); 
         MdcPrntScrn("repetition_time       : %f\n",mr->repetition_time);
         MdcPrntScrn("echo_time             : %g\n",mr->echo_time);
         MdcPrntScrn("inversion_time        : %g\n",mr->inversion_time);
         MdcPrntScrn("num_averages          : %g\n",mr->num_averages);
         MdcPrntScrn("imaging_freq          : %f\n",mr->imaging_freq);
         MdcPrntScrn("pixel_bandwidth       : %g\n",mr->pixel_bandwidth);
         MdcPrntScrn("flip_angle            : %g\n",mr->flip_angle);
         MdcPrntScrn("dbdt                  : %g\n",mr->dbdt);
         MdcPrntScrn("transducer_freq       : %u\n",mr->transducer_freq);
         MdcPrntScrn("transducer_type       : %s\n",mr->transducer_type);
         MdcPrntScrn("pulse_repetition_freq : %u\n",mr->pulse_repetition_freq);
         MdcPrntScrn("pulse_seq_name        : %s\n",mr->pulse_seq_name);
         MdcPrntScrn("steady_state_pulse_seq: %s\n",mr->steady_state_pulse_seq);
         MdcPrntScrn("slab_thickness        : %g\n",mr->slab_thickness);
         MdcPrntScrn("sampling_freq         : %g\n",mr->sampling_freq);
         break;
   }
 }

}


void MdcDebugPrint(char *fmt, ...)
{
  va_list args;

  if (MDC_MY_DEBUG) {
    va_start(args,fmt);
#if GLIBSUPPORTED
    g_logv(MDC_PRGR,G_LOG_LEVEL_DEBUG, fmt, args);
#else
    fprintf(stdout,"\n%s:  Debug : ",MDC_PRGR);
    vsprintf(mdcbufr, fmt, args);
    fprintf(stdout,"%s",mdcbufr);
    fflush(stdout);
#endif
    va_end(args);
  }

}

