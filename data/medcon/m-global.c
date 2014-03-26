/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * filename: m-global.c                                                    *
 *                                                                         *
 * UTIL C-source: Medical Image Conversion Utility                         *
 *                                                                         *
 * purpose      : define global variables                                  *
 *                                                                         *
 * project      : (X)MedCon by Erik Nolf                                   *
 *                                                                         *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* $Id: m-global.c,v 1.80 2010/08/28 23:44:23 enlf Exp $
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
#include "m-defs.h"

/****************************************************************************
                              D E F I N E S 
****************************************************************************/

/* all our version constants */
const char  *MDC_MAJOR = XMEDCON_MAJOR;
const char  *MDC_MINOR = XMEDCON_MINOR;
const char  *MDC_MICRO = XMEDCON_MICRO;
const char  *MDC_PRGR  = XMEDCON_PRGR;
const char  *MDC_DATE  = XMEDCON_DATE;
const char  *MDC_VERSION = XMEDCON_VERSION;
const char  *MDC_LIBVERS = XMEDCON_LIBVERS;

/* fill in host endian and default write endian */
#if MDC_WORDS_BIGENDIAN
Int8 MDC_HOST_ENDIAN = MDC_BIG_ENDIAN;
Int8 MDC_WRITE_ENDIAN= MDC_BIG_ENDIAN;
#else
Int8 MDC_HOST_ENDIAN = MDC_LITTLE_ENDIAN;
Int8 MDC_WRITE_ENDIAN= MDC_LITTLE_ENDIAN;
#endif

Int8 MDC_FILE_ENDIAN = -1;

Int8 MDC_BLOCK_MESSAGES = MDC_NO;

/* globally available variables */
char prefix[MDC_MAX_PREFIX + 1]="eNlf-"; /* 15 + '\O' */
char *mdcbasename=NULL;                  /* new base  */

/* global arrays for argument handling */
char *mdc_arg_files[MDC_MAX_FILES];    /* pointers to filenames         */
int   mdc_arg_convs[MDC_MAX_FRMTS];    /* counter for each conversion   */
int   mdc_arg_total[2];                /* totals for files & conversion */

/* format support compiled in */
Int8 FrmtSupported[MDC_MAX_FRMTS] = {
    0,                  /* MDC_FRMT_NONE  */
    1,                  /* MDC_FRMT_RAW   */
    1,                  /* MDC_FRMT_ASCII */
    MDC_INCLUDE_GIF,    /* MDC_FRMT_GIF   */
    MDC_INCLUDE_ACR,    /* MDC_FRMT_ACR   */
    MDC_INCLUDE_INW,    /* MDC_FRMT_INW   */
    MDC_INCLUDE_ECAT,   /* MDC_FRMT_ECAT6 */
    MDC_INCLUDE_ECAT,   /* MDC_FRMT_ECAT7 */
    MDC_INCLUDE_INTF,   /* MDC_FRMT_INTF  */
    MDC_INCLUDE_ANLZ,   /* MDC_FRMT_ANLZ  */
    MDC_INCLUDE_DICM,   /* MDC_FRMT_DICM  */
    MDC_INCLUDE_PNG,    /* MDC_FRMT_PNG   */
    MDC_INCLUDE_CONC,   /* MDC_FRMT_CONC  */
    MDC_INCLUDE_NIFTI   /* MDC_FRMT_NIFTI */
  /*.................                     */
};
    
/* format name */
char FrmtString[MDC_MAX_FRMTS][15]= {
    "Unknown      ",    /* MDC_FRMT_NONE  */
    "Raw Binary   ",    /* MDC_FRMT_RAW   */
    "Raw Ascii    ",    /* MDC_FRMT_ASCII */
    "Gif89a       ",    /* MDC_FRMT_GIF   */
    "Acr/Nema     ",    /* MDC_FRMT_ACR   */
    "INW (RUG)    ",    /* MDC_FRMT_INW   */
    "CTI ECAT 6   ",    /* MDC_FRMT_ECAT6 */
    "CTI ECAT 7   ",    /* MDC_FRMT_ECAT7 */
    "InterFile    ",    /* MDC_FRMT_INTF  */
    "Analyze      ",    /* MDC_FRMT_ANLZ  */
    "DICOM        ",    /* MDC_FRMT_DICM  */
    "PNG          ",    /* MDC_FRMT_PNG   */
    "Concorde/uPET",    /* MDC_FRMT_CONC  */
    "NIfTI        "     /* MDC_FRMT_NIFTI */
  /*"............."                       */
};

/* format extension */
char FrmtExt[MDC_MAX_FRMTS][8] = {
    "???",    /* MDC_FRMT_NONE  */
    "bin",    /* MDC_FRMT_RAW   */
    "asc",    /* MDC_FRMT_ASCII */
    "gif",    /* MDC_FRMT_GIF   */
    "ima",    /* MDC_FRMT_ACR   */
    "im",     /* MDC_FRMT_INW   */
    "img",    /* MDC_FRMT_ECAT6 */
    "v",      /* MDC_FRMT_ECAT7 */
    "h33",    /* MDC_FRMT_INTF  */
    "hdr",    /* MDC_FRMT_ANLZ  */
    "dcm",    /* MDC_FRMT_DICM  */
    "png",    /* MDC_FRMT_PNG   */
    "img.hdr",/* MDC_FRMT_CONC  */
    "nii"     /* MDC_FRMT_NIFTI */
  /*"..."                       */
};

char mdcbufr[MDC_2KB_OFFSET+1]; /* 2KB global buffer */
char errmsg[MDC_1KB_OFFSET+1];  /* 1KB error  buffer */

/* user specified slope/intercept */
float mdc_si_slope     = 1.;
float mdc_si_intercept = 0.;

/* user specified window center/width */
float mdc_cw_centre    = 0.;
float mdc_cw_width     = 0.;

/* predefined mosaic stamps layout */
Uint32 mdc_mosaic_width = 0;
Uint32 mdc_mosaic_height= 0;
Uint32 mdc_mosaic_number= 0;
Int8   mdc_mosaic_interlaced = MDC_NO;

/* crop settings */
Uint32 mdc_crop_xoffset = 0;
Uint32 mdc_crop_yoffset = 0;
Uint32 mdc_crop_width   = 0;
Uint32 mdc_crop_height  = 0;

/* flags & options */
char MDC_INSTITUTION[MDC_MAXSTR]="NucMed"; /* name of institution           */

Int8 MDC_COLOR_MODE=  MDC_COLOR_RGB;    /* default color mode               */
Int8 MDC_COLOR_MAP =  MDC_MAP_GRAY;     /* gray color palette selected      */

Int8 MDC_PADDING_MODE= MDC_PAD_BOTTOM_RIGHT; /* resized image padding mode */

Int8 MDC_ANLZ_SPM  =  MDC_NO;           /* Analyze/SPM with scaling factor  */
Int8 MDC_ANLZ_OPTIONS = MDC_NO;         /* Analyze/SPM request parameters   */

Int8 MDC_DICOM_MOSAIC_ENABLED = MDC_NO;    /* DICOM: mosaic support enabled   */
Int8 MDC_DICOM_MOSAIC_FORCED  = MDC_NO;    /* DICOM: mosaic preset  forced    */
Int8 MDC_DICOM_MOSAIC_DO_INTERL = MDC_NO;  /* DICOM: mosaic forced interlaced */
Int8 MDC_DICOM_MOSAIC_FIX_VOXEL = MDC_NO;  /* DICOM: mosaic fix voxel sizes   */
Int8 MDC_DICOM_WRITE_IMPLICIT = MDC_NO;    /* DICOM: write little implicit    */
Int8 MDC_DICOM_WRITE_NOMETA = MDC_NO;      /* DICOM: write without meta header*/

Int8 MDC_FORCE_RESCALE = MDC_NO;        /* user specified slope/intercept   */
Int8 MDC_FORCE_CONTRAST= MDC_NO;        /* user specified center/width      */

Int8 MDC_INFO = MDC_YES;                /* default print header info        */
Int8 MDC_INTERACTIVE = MDC_NO;          /* interactive read of raw file     */
Int8 MDC_CONVERT = MDC_NO;              /* image conversion requested       */
Int8 MDC_EXTRACT = MDC_NO;              /* extract images                   */
Int8 MDC_RENAME = MDC_NO;               /* rename base filename             */
Int8 MDC_ECHO_ALIAS = MDC_NO;           /* echo alias name based on ID's    */
Int8 MDC_EDIT_FI = MDC_NO;              /* edit FILEINFO struct             */

Int8 MDC_PIXELS = MDC_NO;               /* print specified pix values       */
Int8 MDC_PIXELS_PRINT_ALL = MDC_NO;     /* print    all    pix values       */

Int8 MDC_NEGATIVE = MDC_NO;             /* allow negative pixel values      */
Int8 MDC_QUANTIFY = MDC_NO;             /* quantitation with one factor     */
Int8 MDC_CALIBRATE = MDC_NO;            /* quantitation with two factors    */
Int8 MDC_CONTRAST_REMAP = MDC_NO;       /* apply contrast remapping         */

Int8 MDC_DEBUG = MDC_NO;                /* give debug info                  */
Int8 MDC_VERBOSE = MDC_NO;              /* run in verbose mode              */

Int8 MDC_GIF_OPTIONS = MDC_NO;          /* request for extra GIF options    */

Int8 MDC_MAKE_GRAY = MDC_NO;            /* forced remap color to gray scale */
Int8 MDC_DITHER_COLOR = MDC_NO;         /* apply dither on color reduction  */

Int8 MDC_NORM_OVER_FRAMES = MDC_NO;     /* normalize over images in a frame */
                                        /* instead of all images            */
Int8 MDC_SKIP_PREVIEW = MDC_NO;         /* skip the first (preview) slice   */
Int8 MDC_IGNORE_PATH = MDC_NO;          /* ignore path in INTF data fname   */
Int8 MDC_SINGLE_FILE = MDC_NO;          /* write INTF as single file        */
Int8 MDC_FORCE_INT = MDC_NO;            /* force integer pixels             */
Int8 MDC_INT16_BITS_USED = 16;          /* bits to use for Int16 type       */
Int8 MDC_TRUE_GAP = MDC_NO;             /* spacing = true gap/overlap       */
Int8 MDC_ALIAS_NAME = MDC_NO;           /* use  alias name based on ID's    */
Int8 MDC_PREFIX_DISABLED = MDC_NO;      /* prevent the prefix in names      */
Int8 MDC_PREFIX_ACQ = MDC_NO;           /* use acquisition number as prefix */
Int8 MDC_PREFIX_SER = MDC_NO;           /* use series      number as prefix */

Int8 MDC_PATIENT_ANON = MDC_NO;         /* make patient anonymous           */
Int8 MDC_PATIENT_IDENT = MDC_NO;        /* give patient identification      */
Int8 MDC_FILE_OVERWRITE = MDC_NO;       /* allow file overwriting           */
Int8 MDC_FILE_STDIN  = MDC_NO;          /* input from stdin  stream         */
Int8 MDC_FILE_STDOUT = MDC_NO;          /* output  to stdout stream         */ 
Int8 MDC_FILE_SPLIT  = MDC_NO;          /* split up file in parts           */
Int8 MDC_FILE_STACK  = MDC_NO;          /* stack up files                   */ 

Int8 MDC_FLIP_HORIZONTAL = MDC_NO;      /* flip horizontal   (x)            */
Int8 MDC_FLIP_VERTICAL   = MDC_NO;      /* flip vertical     (y)            */
Int8 MDC_SORT_REVERSE    = MDC_NO;      /* reverse   sorting                */
Int8 MDC_SORT_CINE_APPLY = MDC_NO;      /* cine apply sorting               */
Int8 MDC_SORT_CINE_UNDO  = MDC_NO;      /* cine undo  sorting               */
Int8 MDC_MAKE_SQUARE     = MDC_NO;      /* make square image                */
Int8 MDC_CROP_IMAGES     = MDC_NO;      /* crop image dimensions            */
Int8 MDC_RESLICE = MDC_NO;              /* reslice images (tra, sag, cor)   */

Int8 MDC_FRMT_INPUT = MDC_FRMT_NONE;    /* format used for stdin            */ 
Int8 MDC_ECAT6_SORT = MDC_ANATOMICAL;   /* ECAT sort order                  */

#if   MDC_INCLUDE_DICM                  /* fallback read format             */
Int8 MDC_FALLBACK_FRMT = MDC_FRMT_DICM;
#elif MDC_INCLUDE_ECAT
Int8 MDC_FALLBACK_FRMT = MDC_FRMT_ECAT6;
#elif MDC_INCLUDE_ANLZ
Int8 MDC_FALLBACK_FRMT = MDC_FRMT_ANLZ;
#elif MDC_INCLUDE_CONC
Int8 MDC_FALLBACK_FRMT = MDC_FRMT_CONC;
#else
Int8 MDC_FALLBACK_FRMT = MDC_FRMT_NONE;
#endif

/* undocumented options, for debugging purposes only */
Int8 MDC_MY_DEBUG=MDC_NO;               /* give even more debug info        */
Int8 MDC_INFO_DB=MDC_NO;                /* just print short database info   */
Int8 MDC_HACK_ACR=MDC_NO;               /* try to find acrnema tags         */

/* XMedCon - GUI*/
Int8 XMDC_GUI=MDC_NO;                   /* is program the GUI part?         */
Int8 XMDC_WRITE_FRMT = MDC_FRMT_RAW;    /* default format to save           */
