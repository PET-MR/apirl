/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * filename: m-progress.c                                                  *
 *                                                                         *
 * UTIL C-source: Medical Image Conversion Utility                         *
 *                                                                         *
 * purpose      : pointer hooks for progress functions                     *
 *                                                                         *
 * project      : (X)MedCon by Erik Nolf                                   *
 *                                                                         *
 * Functions    : MdcSetProgress()   - Set progress value                  *
 *                MdcIncrProgress()  - Increment progress value            *
 *                MdcBeginProgress() - Begin of progress                   *
 *                MdcEndProgress()   - End   of progress                   *
 *                                                                         *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* $Id: m-progress.c,v 1.8 2010/08/28 23:44:23 enlf Exp $
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
#include "m-error.h"
#include "m-progress.h"


/****************************************************************************
                            F U N C T I O N S 
 ****************************************************************************/

int MDC_PROGRESS = MDC_NO;

static void MdcProgressBar(int type, float value, char *label)
{
  switch (type) {
    case MDC_PROGRESS_BEGIN: if (label != NULL) MdcPrntScrn("\n%35s ",label);
        break;
    case MDC_PROGRESS_SET  : MdcPrntScrn(".");
        break;
    case MDC_PROGRESS_INCR : MdcPrntScrn(".");
        break;
    case MDC_PROGRESS_END  : MdcPrntScrn("\n");
        break;
  }
}

void (*MdcProgress)(int type, float value, char *label) = MdcProgressBar;

