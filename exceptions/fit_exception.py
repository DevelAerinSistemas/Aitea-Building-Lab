'''
 # @ Project: AItea-Building-Lab
 # @ Author: Aerin S.L.
 # @ Create Time: 2025-05-29 10:35:08
 # @ Description:
 # @ Version: 1.0.0
 # @ -------:
 # @ Modified by: Aerin S.L.
 # @ Modified time: 2025-05-29 10:35:12
 # @ License: This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY
; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.
'''

class InsufficientDataError(Exception):
    def __init__(self, msg="There is not enough data"):
        self.msg = msg
        super().__init__(self.msg)
    
    def __str__(self):
        return super().__str__()
