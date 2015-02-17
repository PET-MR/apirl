/******************************************************************************

  Copyright (c) 2005,2009 Turku PET Centre

  Library:      interfile
  Description:  Function(s) for interfile headers

  This program is free software; you can redistribute it and/or modify it under
  the terms of the GNU General Public License as published by the Free Software
  Foundation; either version 2 of the License, or (at your option) any later
  version.

  This program is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
  FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License along with
  this program; if not, write to the Free Software Foundation, Inc., 59 Temple
  Place, Suite 330, Boston, MA 02111-1307 USA.

  Turku PET Centre hereby disclaims all copyright interest in the program.

  Juhani Knuuti
  Director, Professor
  Turku PET Centre, Turku, Finland, http://www.turkupetcentre.fi

  Modification history:
   2005-04-05 version 1.0 (krs) Roman Krais
   2009-02-26 VO
     fread() cast with (void) to prevent compiler warnings.
   2009-03-04 VO
     fread() return value is verified.

*/

#include <stdio.h>
#include <string.h>

#include <Interfile.h>

/*!
 * The function searches the keyword in the header and passes the value
 * belonging to that value back to the main program.
 * The name of the header (string 'headerName') and the requested keyword
 * (string 'searchWord') are passed to the function. It passes back the
 * value of the keyword (string 'returnValue') and possibly an error message
 * or warning (string 'errorMessage'). So the values are passed back as strings.
 * The interpretation (conversion to integer, float, time etc) is up
 * to the programmer.
 *
 * The interfile header has to comply to the following rules:
 * - first line in the file is '!INTERFILE'
 * - maximal length of a line is 512 characters
 * - A line has two fields sperated by ':=' (keyword := value)
 * - maximal length of keyword and value is 256 characters.
 * - no header entries after a line  '!END OF INTERFILE'
 * - a line starting with a semicolon ';' is a comment
 *
 * @param headerName header file name
 * @param searchWord keyword to look for
 * @param returnValue value for keyword in header
 * @param errorMessage error message/warnings. In case there is a error message it will be returnd as string in the
 * variable 'errmsg'.
 * @return 0 if ok, 1  keyword appears more than once in the interfile header
 * (value of last occurence of keyword is returned), 
 * 2 keyword not found in interfile header (returned value is empty (i.e. contains '/0's only)),
 * 3  interfile header cold not be opened for reading (returned value is empty (i.e. contains '/0's only)),
 * 4  wrong file format?! (No '!INTERFILE' in the first line) (returned value is empty (i.e. contains '/0's only))
 */
int interfile_read(char* headerName, char* searchWord, char* returnValue, char* errorMessage) {
  short int  i, pos;
  short int  count=0;    /* counter: How often appears keyword in the header? */
  int        n;
  char       c[1];
  char       keyword[1024], value[1024];
  char       line[512];  /* max length of a line accepted in interfile header */
  FILE       *interfileHeader;

  /* initialise strings */
  returnValue[0] = '\0';
  errorMessage[0] = '\0';

                                         /* open interfile header for reading */
  if ((interfileHeader = fopen(headerName,"r"))==NULL) {
    strcpy(errorMessage,headerName);
    strcat(errorMessage," could not be opened for reading");
    return 3;
  }

                  /* check from first line if file is really interfile header */
  n=fread(&c,1,1,interfileHeader); if(n<1) {
    strcpy(errorMessage,"wrong file header format?! No '!INTERFILE' at start of ");
    strcat(errorMessage,headerName);
    fclose(interfileHeader);
    return 4;
  }
  i=0;
  memcpy(&line[i],c,1);
  while (memcmp(c,"\n",1) && memcmp(c,"\r",1)) {
    i++;
    n=fread(&c,1,1,interfileHeader); if(n<1) {
      strcpy(errorMessage,"wrong file header format?! No '!INTERFILE' at start of ");
      strcat(errorMessage,headerName);
      fclose(interfileHeader);
      return 4;
    }
    memcpy(&line[i],c,1);
  }
  if (memcmp(line,"!INTERFILE",10)) {
    strcpy(errorMessage,"wrong file header format?! No '!INTERFILE' at start of ");
    strcat(errorMessage,headerName);
    fclose(interfileHeader);
    return 4;
  }

                                                    /* read file line by line */
 while (fread(&c,1,1,interfileHeader) == 1) {
    for (i=0;i<516;i++) line[i] = '\0';                    /* initialise line */
    for (i=0;i<256;i++) keyword[i] = '\0';              /* initialise keyword */
    for (i=0;i<256;i++) value[i] = '\0';                  /* initialise value */
    i=0;
             /* \n = end of line, \r = carriage return. Lines in  ASCII files */
             /* on Sun-Solaris end with \n, on Intel-Windows with \r\n        */
    //while (memcmp(c,"\r",1) && memcmp(c,"\n",1) && i<516) {
    while (memcmp(c,"\n",1) && i<1024) {	// En linux es solo \n
      memcpy(&line[i],c,1);
      n=fread(&c,1,1,interfileHeader); if(n<1) {
        strcpy(errorMessage,"wrong file header format: ");
        strcat(errorMessage,headerName);
        fclose(interfileHeader);
        return 4;
      }
      i++;
    }
                                                /* comments are not processed */
    if (strncmp(&line[0],";",1)) {
                                           /* get keyword and value from line */
                                 /* find position of the field seperator ':=' */
      for (pos=1; pos<516; pos++)
        if (line[pos] == '=' && line[pos-1] == ':') break; 
                                    /* now get the first and the second field */
      for (i=0;i<pos-2 && i<256;i++) keyword[i] = line[i];
      for (i=pos+2;i<256+pos+2 && i<1024;i++) {
        if (!memcmp(&line[i],"\0",1) || !memcmp(&line[i],"\r",1) || !memcmp(&line[i],"\n",1)) 
          break;                                 /* stop at the end of "line" */
        value[i-pos-2] = line[i];
      }
      if (!memcmp(keyword,"!END OF INTERFILE",17)) break;     /* are we done? */
                                             /* check if we found the keyword */
       else if (!strcmp(keyword,searchWord)) {
              strcpy(returnValue,value);
              count++;
            }
    }
  }
  fclose(interfileHeader);                               /* done with reading */
  if (count == 0) {
    strcpy(errorMessage,"keyword '");
    strcat(errorMessage,searchWord);
    strcat(errorMessage,"' not found in header");
    return 2;
  }
  if (count > 1) {
    strcpy(errorMessage,"keyword '");
    strcat(errorMessage,searchWord);
    strcat(errorMessage,"' appears more than once in header");
    return 1;
  }
  return 0;
}
