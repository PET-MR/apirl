/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * filename: m-defs.h                                                      *
 *                                                                         *
 * UTIL C-source: Medical Image Conversion Utility                         *
 *                                                                         *
 * purpose      : project variables, structs & datatypes                   *
 *                                                                         *
 * project      : (X)MedCon by Erik Nolf                                   *
 *                                                                         *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* $Id: m-defs.h,v 1.57 2010/08/28 23:44:23 enlf Exp $
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

#include "m-config.h"

/**************************
  COMPILATION CONDITIONALS
 **************************/

#ifdef _WIN32
/* dos/mswindows */
#define MDC_PATH_DELIM_STR "\\"
#define MDC_PATH_DELIM_CHR '\\'
#define MDC_NEWLINE "\r\n"
#else
/* unix/linux */
#define MDC_PATH_DELIM_STR "/"
#define MDC_PATH_DELIM_CHR '/'
#define MDC_NEWLINE "\n"
#endif

/* For output formats without slice_spacing key, it is prefered
 * to use slice_spacing as the slice_width parameter, and thus
 * virtually "eliminating" any gaps/overlaps between those sices.
 * If not wanted, comment out following define line.
 */
//#define MDC_USE_SLICE_SPACING 1


/*******************
  PORTABILITY TYPES
 *******************/
/* 16 bit type */
#if     (MDC_SIZEOF_INT   == 2)
# define INT_2BYTE
#elif   (MDC_SIZEOF_SHORT == 2)
# define SHORT_2BYTE
#else
# error "What!?  No 16-bit integer type available!"
#endif
/* 32 bit type */
#if     (MDC_SIZEOF_INT   == 4)
# define INT_4BYTE
#elif   (MDC_SIZEOF_LONG  == 4)
# define LONG_4BYTE
#else
# error "What!?  No 32-bit integer type available!"
#endif
/* 64 bit type */
#ifdef MDC_SIZEOF_LONG_LONG
#if     (MDC_SIZEOF_LONG_LONG == 8)
# define LONG_LONG_8BYTE
#endif
#else
#if     (MDC_SIZEOF_LONG == 8)
# define LONG_8BYTE
#endif
#endif


/****************************************************************************
                              D E F I N E S
****************************************************************************/

#define MDC_ONE           1
#define MDC_ZERO          0

#define MDC_FALSE        (0)
#define MDC_TRUE         (!MDC_FALSE)

#define MDC_UNKNOWN      MDC_ZERO

#define MDC_LITTLE_ENDIAN  MDC_ONE
#define MDC_BIG_ENDIAN     MDC_ZERO

#define MDC_1KB_OFFSET     1024
#define MDC_2KB_OFFSET     2048

#define MDC_INPUT_NORM_STYLE   1
#define MDC_INPUT_ECAT_STYLE   2

#define MDC_MAX_PATH  256
#define MDC_MAX_LIST  256

#define MDC_YES    MDC_ONE
#define MDC_NO     MDC_ZERO

#define MDC_ARG_FILE    0
#define MDC_ARG_CONV    1
#define MDC_ARG_EXTRACT 2

#define MDC_FILES     0
#define MDC_CONVS     1

/* precision */
#define MDC_FLT_EPSILON 1.1920928955078125e-07

/* passess */
#define MDC_PASS0     0
#define MDC_PASS1     1
#define MDC_PASS2     2

/* output message levels */
#define MDC_LEVEL_MESG    1
#define MDC_LEVEL_WARN    2
#define MDC_LEVEL_ERR     3
#define MDC_LEVEL_ALL     4

/* supported color maps */
#define MDC_MAP_PRESENT   MDC_ZERO /* 256 RGB   colormap */
#define MDC_MAP_GRAY      1        /* grayscale colormap */
#define MDC_MAP_INVERTED  2        /* inverted  colormap */
#define MDC_MAP_RAINBOW   3        /* rainbow   colormap */
#define MDC_MAP_COMBINED  4        /* combined  colormap */
#define MDC_MAP_HOTMETAL  5        /* hotmetal  colormap */

#define MDC_MAP_LOADED    6        /* loaded    colormap */

/* supported color modes */
#define MDC_COLOR_INDEXED 0        /* 256 indexed colors */
#define MDC_COLOR_RGB     1        /* 24bit true  colors */

/* supported formats */
#define MDC_FRMT_BAD     MDC_ZERO
#define MDC_FRMT_NONE    MDC_FRMT_BAD
#define MDC_FRMT_RAW     1
#define MDC_FRMT_ASCII   2
#define MDC_FRMT_GIF     3
#define MDC_FRMT_ACR     4
#define MDC_FRMT_INW     5
#define MDC_FRMT_ECAT6   6
#define MDC_FRMT_ECAT7   7
#define MDC_FRMT_INTF    8
#define MDC_FRMT_ANLZ    9
#define MDC_FRMT_DICM    10
#define MDC_FRMT_PNG     11
#define MDC_FRMT_CONC    12
#define MDC_FRMT_NIFTI   13

#define MDC_MAX_FRMTS    14        /* total+1 conversion formats supported */

/* acquisition types */                      /*  InterFile and DICOM   */
#define MDC_ACQUISITION_UNKNOWN    MDC_ZERO  /* unknown = static       */
#define MDC_ACQUISITION_STATIC     1         /* static, simple default */
#define MDC_ACQUISITION_DYNAMIC    2         /* dynamic                */
#define MDC_ACQUISITION_TOMO       3         /* tomographic            */
#define MDC_ACQUISITION_GATED      4         /* gated                  */
#define MDC_ACQUISITION_GSPECT     5         /* gated spect            */
/// Agregado por Martin belzunce:
#define MDC_ACQUISITION_PET        6         /* PET            */
/// Modificado de 6 a 7 por Martin belzunce:
#define MDC_MAX_ACQUISITIONS       7         /* total acquisitions + 1 */

/* ECAT sort order */
#define MDC_ANATOMICAL   1
#define MDC_BYFRAME      2

/* pat_orient */
#define MDC_LEFT         1
#define MDC_RIGHT        2
#define MDC_ANTERIOR     3
#define MDC_POSTERIOR    4
#define MDC_HEAD         5
#define MDC_FEET         6

/* patient rotation */
#define MDC_SUPINE            1       /* on the back            */
#define MDC_PRONE             2       /* on the face            */
#define MDC_DECUBITUS_RIGHT   3       /* on the right side      */
#define MDC_DECUBITUS_LEFT    4       /* on the left side       */

/* patient orientation */
#define MDC_HEADFIRST   1             /* head first in scanner  */
#define MDC_FEETFIRST   2             /* feet first in scanner  */

/* slice orientation */               /* consider a patient on  */
                                      /* his back on the table, */
                                      /* then the direction is: */
#define MDC_TRANSAXIAL  1             /* //  device ;_|_ ground */
#define MDC_SAGITTAL    2             /* _|_ device ;_|_ ground */
#define MDC_CORONAL     3             /* _|_ device ; // ground */

/* patient/slice combined */
#define MDC_SUPINE_HEADFIRST_TRANSAXIAL           1
#define MDC_SUPINE_HEADFIRST_SAGITTAL             2
#define MDC_SUPINE_HEADFIRST_CORONAL              3
#define MDC_SUPINE_FEETFIRST_TRANSAXIAL           4
#define MDC_SUPINE_FEETFIRST_SAGITTAL             5
#define MDC_SUPINE_FEETFIRST_CORONAL              6
#define MDC_PRONE_HEADFIRST_TRANSAXIAL            7
#define MDC_PRONE_HEADFIRST_SAGITTAL              8
#define MDC_PRONE_HEADFIRST_CORONAL               9
#define MDC_PRONE_FEETFIRST_TRANSAXIAL           10
#define MDC_PRONE_FEETFIRST_SAGITTAL             11
#define MDC_PRONE_FEETFIRST_CORONAL              12
#define MDC_DECUBITUS_RIGHT_HEADFIRST_TRANSAXIAL 13
#define MDC_DECUBITUS_RIGHT_HEADFIRST_SAGITTAL   14
#define MDC_DECUBITUS_RIGHT_HEADFIRST_CORONAL    15
#define MDC_DECUBITUS_RIGHT_FEETFIRST_TRANSAXIAL 16
#define MDC_DECUBITUS_RIGHT_FEETFIRST_SAGITTAL   17
#define MDC_DECUBITUS_RIGHT_FEETFIRST_CORONAL    18
#define MDC_DECUBITUS_LEFT_HEADFIRST_TRANSAXIAL  19
#define MDC_DECUBITUS_LEFT_HEADFIRST_SAGITTAL    20
#define MDC_DECUBITUS_LEFT_HEADFIRST_CORONAL     21
#define MDC_DECUBITUS_LEFT_FEETFIRST_TRANSAXIAL  22
#define MDC_DECUBITUS_LEFT_FEETFIRST_SAGITTAL    23
#define MDC_DECUBITUS_LEFT_FEETFIRST_CORONAL     24
#define MDC_MAX_ORIENT                           25 /* total orientations + 1 */

/* detector rotation */
#define MDC_ROTATION_CW   1   /* clockwise               */
#define MDC_ROTATION_CC   2   /* counter-clocwise        */

/* detector motion   */
#define MDC_MOTION_STEP   1   /* stepped                 */
#define MDC_MOTION_CONT   2   /* continuous              */
#define MDC_MOTION_DRNG   3   /* during step             */

/* gated spect nesting outer level */
#define MDC_GSPECT_NESTING_SPECT 1
#define MDC_GSPECT_NESTING_GATED 2

/* gated heart rate */
#define MDC_HEART_RATE_ACQUIRED  1
#define MDC_HEART_RATE_OBSERVED  2

/* image padding */
#define MDC_PAD_AROUND        1
#define MDC_PAD_TOP_LEFT      2
#define MDC_PAD_BOTTOM_RIGHT  3

/* some maximum limits */
#define MDC_MAX_FILES 10000 /* maximum files handled        */
                            /* 3-char prefix: 34696 uniques */

#define MDC_MAXSTR      35  /* max length of strings        */

#define MDC_BUF_ITMS    10  /* realloc per BUF_ITMS items   */

#define MDC_CHAR_BUF   100  /* max chars for string buffer  */

#define MDC_MAX_PREFIX  15  /* max chars for prefix         */

/* pixel types                             */
#define BIT1      1    /*  1-bit           */
#define BIT8_S    2    /*  8-bit signed    */
#define BIT8_U    3    /*  8-bit unsigned  */
#define BIT16_S   4    /* 16-bit signed    */
#define BIT16_U   5    /* 16-bit unsigned  */
#define BIT32_S   6    /* 32-bit signed    */
#define BIT32_U   7    /* 32-bit unsigned  */
#define BIT64_S   8    /* 64-bit signed    */
#define BIT64_U   9    /* 64-bit unsigned  */
#define FLT32    10    /* 32-bit float     */
#define FLT64    11    /* 64-bit double    */
#define ASCII    12    /* ascii            */
#define VAXFL32  13    /* 32-bit vaxfloat  */
#define COLRGB   20    /* RGB triplets     */

/* define maximum integer values */
#define MDC_MAX_BIT8_U    255
#define MDC_MAX_BIT16_S   ((1<<16)/2) - 1

/* file compression type */
#define MDC_COMPRESS  1
#define MDC_GZIP      2

/*  8 bit type */
typedef   signed char Int8;
typedef unsigned char Uint8;

/* 16 bit type */
#ifdef SHORT_2BYTE
 typedef   signed short Int16;
 typedef unsigned short Uint16;
#elif  INT_2BYTE
 typedef   signed int  Int16;
 typedef unsigned int Uint16;
#endif

/* 32 bit type */
#ifdef INT_4BYTE
 typedef   signed int  Int32;
 typedef unsigned int Uint32;
#elif  LONG_4BYTE
 typedef   signed long Int32;
 typedef unsigned long Uint32;
#endif

/* 64 bit type */
#ifdef LONG_8BYTE
#define HAVE_8BYTE_INT
 typedef   signed long Int64;
 typedef unsigned long Uint64;
#else
#ifdef LONG_LONG_8BYTE
#define HAVE_8BYTE_INT
 typedef   signed long long Int64;
 typedef unsigned long long Uint64;
#endif
#endif

/* define different modalities */
typedef enum {

  M_AS=('A'<<8)|'S',    /* Angioscopy                                 */
  M_AU=('A'<<8)|'U',    /* Audio                                      */
  M_BI=('B'<<8)|'I',    /* Biomagnetic Imaging                        */
  M_CD=('C'<<8)|'D',    /* Color Flow Doppler                         */
  M_CF=('C'<<8)|'F',    /* Cinefluorography                           */
  M_CP=('C'<<8)|'P',    /* Culposcopy                                 */
  M_CR=('C'<<8)|'R',    /* Computed Radiography                       */
  M_CS=('C'<<8)|'S',    /* Cystoscopy                                 */
  M_CT=('C'<<8)|'T',    /* Computed Tomography                        */
  M_DD=('D'<<8)|'D',    /* Duplex Doppler                             */
  M_DF=('D'<<8)|'F',    /* Digital Fluoroscopy                        */
  M_DG=('D'<<8)|'G',    /* Diaphanography                             */
  M_DM=('D'<<8)|'M',    /* Digital Microscopy                         */
  M_DS=('D'<<8)|'S',    /* Digital Substraction Angiography           */
  M_DX=('D'<<8)|'X',    /* Digital Radiography                        */
  M_EC=('E'<<8)|'C',    /* Echocardiography                           */
  M_ES=('E'<<8)|'S',    /* Endoscopy                                  */
  M_FA=('F'<<8)|'A',    /* Fluorescein Angiography                    */
  M_FS=('F'<<8)|'S',    /* Fundoscopy                                 */
  M_GM=('G'<<8)|'M',    /* General Microscopy                         */
  M_HD=('H'<<8)|'D',    /* Hemodynamic Waveform                       */
  M_IO=('I'<<8)|'O',    /* Intra-Oral Radiography                     */
  M_HC=('H'<<8)|'C',    /* Hardcopy                                   */
  M_LP=('L'<<8)|'P',    /* Laparoscopy                                */
  M_MA=('M'<<8)|'A',    /* Magnetic Resonance Angiography             */
  M_MG=('M'<<8)|'G',    /* Mammography                                */
  M_MR=('M'<<8)|'R',    /* Magnetic Resonance                         */
  M_MS=('M'<<8)|'S',    /* Magnetic Resonance Spectroscopy            */
  M_NM=('N'<<8)|'M',    /* Nuclear Medicine                           */
  M_OT=('O'<<8)|'T',    /* Other                                      */
  M_PT=('P'<<8)|'T',    /* Positron Emission Tomography               */
  M_PX=('P'<<8)|'X',    /* Panoramic X-Ray                            */
  M_RF=('R'<<8)|'F',    /* Radio Fluoroscopy                          */
  M_RG=('R'<<8)|'G',    /* Radiographic Imaging                       */
  M_RT=('R'<<8)|'T',    /* Radiotherapy                               */
  M_SM=('S'<<8)|'M',    /* Slide Microscopy                           */
  M_SR=('S'<<8)|'R',    /* SR Document                                */
  M_ST=('S'<<8)|'T',    /* Single-Photon Emission Computed Tomography */
  M_TG=('T'<<8)|'G',    /* Thermography                               */
  M_US=('U'<<8)|'S',    /* Ultrasound                                 */
  M_VF=('V'<<8)|'F',    /* Videofluorography                          */
  M_XA=('X'<<8)|'A',    /* X-Ray Angiography                          */
  M_XC=('X'<<8)|'C'     /* External-Camera Photography                */

}
MDC_MODALITY;
