/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * filename: medcon.h                                                      *
 *                                                                         *
 * UTIL C-source: Medical Image Conversion Utility                         *
 *                                                                         *
 * purpose      : project header file                                      *
 *                                                                         *
 * project      : (X)MedCon by Erik Nolf                                   *
 *                                                                         *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* $Id: medcon.h,v 1.35 2010/08/28 23:44:23 enlf Exp $
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
 
#ifndef __MEDCON_H__
#define __MEDCON_H__

/****************************************************************************
                              H E A D E R S
****************************************************************************/

#ifdef __cplusplus
extern "C" {
#endif

/* define _WIN32 in case of cygwin/mingw compilation */
#ifndef _WIN32
#  ifdef __MINGW32__
#    define _WIN32
#  elif defined __CYGWIN32__
#    define _WIN32
#  endif
#endif

#include <stdio.h>

#include "m-defs.h"
#include "m-structs.h"
#include "m-error.h"
#include "m-files.h"
#include "m-debug.h"
#include "m-fancy.h"
//#include "m-getopt.h"
#include "m-algori.h"
#include "m-color.h"
#include "m-global.h"
//#include "m-pixels.h"
//#include "m-xtract.h"
//#include "m-init.h"
//#include "m-vifi.h"
//#include "m-rslice.h"
#include "m-transf.h"
//#include "m-qmedian.h"
#include "m-split.h"
#include "m-stack.h"
#include "m-progress.h"

#include "m-raw.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#if MDC_INCLUDE_GIF
#  include "m-gif.h"
#endif
#if MDC_INCLUDE_ACR
#  include "m-acr.h"
#endif
#if MDC_INCLUDE_INW
#  include "m-inw.h"
#endif
#if MDC_INCLUDE_INTF
#  include "m-intf.h"
#endif
#if MDC_INCLUDE_CONC
#  include "m-conc.h"
#endif
#if MDC_INCLUDE_ECAT
#  include "m-ecat64.h"
#  include "m-ecat72.h"
#endif
#if MDC_INCLUDE_ANLZ
#  include "m-anlz.h"
#endif
#if MDC_INCLUDE_DICM
#  include "m-dicm.h"
#endif
#if MDC_INCLUDE_PNG
#  include "m-png.h"
#endif
#if MDC_INCLUDE_NIFTI
#  include "m-nifti.h"
#endif

#ifdef __cplusplus
}
#endif

#endif

