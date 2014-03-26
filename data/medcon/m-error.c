/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * filename: m-error.c                                                     *
 *                                                                         *
 * UTIL C-source: Medical Image Conversion Utility                         *
 *                                                                         *
 * purpose      : Handle warnings and errors                               *
 *                                                                         *
 * project      : (X)MedCon by Erik Nolf                                   *
 *                                                                         *
 * Functions    : MdcPrntStream()       - Gives proper output stream       *
 *                MdcPrntScrn()         - Print to screen                  *
 *                MdcPrntMesg()         - Print a message                  *
 *                MdcPrntWarn()         - Print a warning                  *
 *                MdcPrntErr()          - Print error and leave            *
 *                                                                         *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* $Id: m-error.c,v 1.26 2010/08/28 23:44:23 enlf Exp $
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
#include <stdarg.h>
#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif

#include "m-defs.h"
#include "m-global.h"
#include "m-error.h"

#if GLIBSUPPORTED
#include <glib.h>
#endif

/****************************************************************************
                            F U N C T I O N S
****************************************************************************/

static FILE *MdcPrntStream(void)
{ 
  FILE *stream;

  if (MDC_FILE_STDOUT == MDC_YES) {
    stream = stderr;
  }else{
    stream = stdout;
  }
  return(stream);
}


void MdcPrntScrn(char *fmt, ...)
{
  va_list args;

  va_start(args, fmt);
  vfprintf(MdcPrntStream(), fmt, args);
  va_end(args);

}

void MdcPrntMesg(char *fmt, ...)
{
  va_list args;

  if (MDC_BLOCK_MESSAGES >= MDC_LEVEL_MESG) return;

  va_start(args,fmt);

#if GLIBSUPPORTED
  g_logv(MDC_PRGR,G_LOG_LEVEL_MESSAGE, fmt, args);
#else
  MdcPrntScrn("\n%s: Message: ",MDC_PRGR);
  vfprintf(MdcPrntStream(), fmt, args); fprintf(MdcPrntStream(),"\n\n");
#endif

  va_end(args);

}

void MdcPrntWarn(char *fmt, ...)
{
  va_list args; 

  if (MDC_BLOCK_MESSAGES >= MDC_LEVEL_WARN) return;

  va_start(args, fmt);

#if GLIBSUPPORTED
  g_logv(MDC_PRGR,G_LOG_LEVEL_WARNING, fmt, args);
#else
  MdcPrntScrn("\n%s: Warning: ",MDC_PRGR);
  vfprintf(MdcPrntStream(), fmt, args); fprintf(MdcPrntStream(),"\n\n");
#endif

  va_end(args);

}

void MdcPrntErr(int code, char *fmt, ...)
{
  va_list args;

  if (MDC_BLOCK_MESSAGES >= MDC_LEVEL_ERR) exit(-code);

  va_start(args, fmt);

#if GLIBSUPPORTED
  g_logv(MDC_PRGR,G_LOG_LEVEL_ERROR, fmt, args);
#else
  MdcPrntScrn("\n%s: Error  : ",MDC_PRGR);
  vfprintf(MdcPrntStream(), fmt, args); fprintf(MdcPrntStream(),"\n\n");
#endif

  va_end(args);

  exit(-code);

}

