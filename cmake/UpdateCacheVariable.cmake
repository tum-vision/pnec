#
# This file is part of iRotAvg.
# This file is based on the file UpdateCacheVariable.cmake
# from Ceres Solver (released under the New BSD license,
# see its oroginal header below).
#
# Created by Alvaro Parra on 19/3/19.
# Copyright © 2019 Alvaro Parra <alvaro dot parrabustos at adelaide
# dot edu dot au> (The University of Adelaide)
# For more information see <https://github.com/ajparra/iRotAvg>
#
# This work was supported by Maptek (http://maptek.com) and the
# ARC Linkage Project LP140100946.
#
# iRotAvg is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# iRotAvg is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with iRotAvg. If not, see <http://www.gnu.org/licenses/>.
#

#
############################################################################
# Ceres Solver - A fast non-linear least squares minimizer
# Copyright 2015 Google Inc. All rights reserved.
# http://ceres-solver.org/
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of Google Inc. nor the names of its contributors may be
#   used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: alexs.mac@gmail.com (Alex Stewart)

# By default, there is no easy way in CMake to set the value of a cache
# variable without reinitialising it, which involves resetting its
# associated help string.  This is particularly annoying for CMake options
# where they need to programmatically updated.
#
# This function automates this process by getting the current help string
# for the cache variable to update, then reinitialising it with the new
# value, but with the original help string.
function(UPDATE_CACHE_VARIABLE VAR_NAME VALUE)
  get_property(IS_DEFINED_IN_CACHE CACHE ${VAR_NAME} PROPERTY VALUE SET)
  if (NOT IS_DEFINED_IN_CACHE)
    message(FATAL_ERROR "Specified variable to update in cache: "
      "${VAR_NAME} has not been set in the cache.")
  endif()
  get_property(HELP_STRING CACHE ${VAR_NAME} PROPERTY HELPSTRING)
  get_property(VAR_TYPE CACHE ${VAR_NAME} PROPERTY TYPE)
  set(${VAR_NAME} ${VALUE} CACHE ${VAR_TYPE} "${HELP_STRING}" FORCE)
endfunction()

