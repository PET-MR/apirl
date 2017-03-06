/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * filename: m-error.h                                                     *
 *                                                                         *
 * UTIL C-source: Medical Image Conversion Utility                         *
 *                                                                         *
 * purpose      : m-error.c header file                                    *
 *                                                                         *
 * project      : (X)MedCon by Erik Nolf                                   *
 *                                                                         *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* $Id: m-error.h,v 1.13 2010/08/28 23:44:23 enlf Exp $
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

#define MDC_OK               0       /* return codes */
#define MDC_BAD_OPEN        -1
#define MDC_BAD_CLOSE       -2
#define MDC_BAD_FILE        -3
#define MDC_BAD_READ        -4
#define MDC_UNEXPECTED_EOF  -5
#define MDC_BAD_CODE        -6
#define MDC_BAD_FIRSTCODE   -7
#define MDC_BAD_ALLOC       -8
#define MDC_BAD_SYMBOLSIZE  -9
#define MDC_OVER_FLOW       -10
#define MDC_NO_CODE         -11
#define MDC_BAD_WRITE       -12

/****************************************************************************
                            F U N C T I O N S
****************************************************************************/

void MdcPrntScrn(char *fmt, ...);
void MdcPrntWarn(char *fmt, ...);
void MdcPrntMesg(char *fmt, ...);
void MdcPrntErr(int code, char *fmt, ...);
