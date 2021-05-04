'''
Copyright (c) 2019-2020 Erkin Ötleş. ALL RIGHTS RESERVED.

Unauthorized duplication and/or distribution prohibited. Proprietary and confidential.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

def string_to_time(time):
    if isinstance(time, int) or isinstance(time, float):
        return time
    
    numbers = [int(word) for word in time.split() if word.isdigit()]
    number = numbers[0]
    if "second" in time:
        return number
    if "min" in time:
        return number*60
    if "h" in time:
        return number*60*60
    if "day" in time:
        return number*86400
    if "w" in time:
        return number*86400*7
    if "mon" in time:
        return number*86400*30