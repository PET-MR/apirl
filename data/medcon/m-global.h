/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * filename: m-global.h                                                    *
 *                                                                         *
 * UTIL C-source: Medical Image Conversion Utility                         *
 *                                                                         *
 * purpose      : m-global.c header file                                   *
 *                                                                         *
 * project      : (X)MedCon by Erik Nolf                                   *
 *                                                                         *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* $Id: m-global.h,v 1.65 2010/08/28 23:44:23 enlf Exp $
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
****************************************************************************/

extern const char *MDC_MAJOR;
extern const char *MDC_MINOR;
extern const char *MDC_MICRO;
extern const char *MDC_PRGR;
extern const char *MDC_DATE;
extern const char *MDC_VERSION;
extern const char *MDC_LIBVERS;

extern char mdcbufr[MDC_2KB_OFFSET+1];
extern char errmsg[MDC_1KB_OFFSET+1];
extern char prefix[MDC_MAX_PREFIX+1];
extern char *mdcbasename;

extern char *mdc_arg_files[MDC_MAX_FILES];
extern int   mdc_arg_convs[MDC_MAX_FRMTS];
extern int   mdc_arg_total[2];

extern Int8 FrmtSupported[MDC_MAX_FRMTS];
extern char FrmtString[MDC_MAX_FRMTS][15];
extern char FrmtExt[MDC_MAX_FRMTS][8];

extern float mdc_si_slope;
extern float mdc_si_intercept;
extern float mdc_cw_centre;
extern float mdc_cw_width;

extern Uint32 mdc_mosaic_width;
extern Uint32 mdc_mosaic_height;
extern Uint32 mdc_mosaic_number;
extern Int8   mdc_mosaic_interlaced;

extern Uint32 mdc_crop_xoffset;
extern Uint32 mdc_crop_yoffset;
extern Uint32 mdc_crop_width;
extern Uint32 mdc_crop_height;

extern char MDC_INSTITUTION[MDC_MAXSTR];

extern Int8 MDC_COLOR_MODE;
extern Int8 MDC_COLOR_MAP;
extern Int8 MDC_PADDING_MODE;
extern Int8 MDC_ANLZ_SPM, MDC_ANLZ_OPTIONS;
extern Int8 MDC_DICOM_MOSAIC_ENABLED, MDC_DICOM_MOSAIC_FORCED;
extern Int8 MDC_DICOM_MOSAIC_DO_INTERL, MDC_DICOM_MOSAIC_FIX_VOXEL;
extern Int8 MDC_DICOM_WRITE_IMPLICIT, MDC_DICOM_WRITE_NOMETA;
extern Int8 MDC_FORCE_RESCALE;
extern Int8 MDC_FORCE_CONTRAST;
extern Int8 MDC_HOST_ENDIAN, MDC_FILE_ENDIAN;
extern Int8 MDC_BLOCK_MESSAGES;

extern Int8 MDC_INFO, MDC_INTERACTIVE, MDC_CONVERT;
extern Int8 MDC_EXTRACT, MDC_NEGATIVE;
extern Int8 MDC_PIXELS, MDC_PIXELS_PRINT_ALL;
extern Int8 MDC_QUANTIFY, MDC_CALIBRATE, MDC_DEBUG;
extern Int8 MDC_CONTRAST_REMAP;
extern Int8 MDC_GIF_OPTIONS;
extern Int8 MDC_MAKE_GRAY, MDC_DITHER_COLOR;
extern Int8 MDC_VERBOSE, MDC_RENAME;
extern Int8 MDC_NORM_OVER_FRAMES;
extern Int8 MDC_SKIP_PREVIEW, MDC_IGNORE_PATH, MDC_SINGLE_FILE;
extern Int8 MDC_FORCE_INT;
extern Int8 MDC_INT16_BITS_USED;
extern Int8 MDC_TRUE_GAP;
extern Int8 MDC_ALIAS_NAME;
extern Int8 MDC_ECHO_ALIAS;
extern Int8 MDC_PREFIX_DISABLED, MDC_PREFIX_ACQ, MDC_PREFIX_SER;
extern Int8 MDC_RESLICE;
extern Int8 MDC_PATIENT_ANON, MDC_PATIENT_IDENT;
extern Int8 MDC_EDIT_FI;
extern Int8 MDC_FILE_OVERWRITE;
extern Int8 MDC_FILE_STDOUT;
extern Int8 MDC_FILE_STDIN;
extern Int8 MDC_FILE_SPLIT, MDC_FILE_STACK;
extern Int8 MDC_FLIP_HORIZONTAL, MDC_FLIP_VERTICAL;
extern Int8 MDC_SORT_REVERSE, MDC_SORT_CINE_APPLY, MDC_SORT_CINE_UNDO;
extern Int8 MDC_MAKE_SQUARE, MDC_CROP_IMAGES;
extern Int8 MDC_FRMT_INPUT;
extern Int8 MDC_WRITE_ENDIAN;
extern Int8 MDC_MY_DEBUG;
extern Int8 MDC_INFO_DB;
extern Int8 MDC_HACK_ACR;
extern Int8 MDC_ECAT6_SORT;
extern Int8 MDC_FALLBACK_FRMT;


extern char *mdc_comments;

extern Int8 XMDC_GUI, XMDC_WRITE_FRMT;
