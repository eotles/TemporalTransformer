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


import sqlite3

class cursor_manager():
    def __init__(self, cur, verbose=True):
        self.cur = cur
        self.verbose = verbose
        
    def interactive_session(self):
        interactive_session(self.cur)
    
    def execute_sql(self, sql_stmt, param=None, many=False):
        execute_sql(self.cur, sql_stmt,
                    verbose=self.verbose, param=param, many=many)
    
    def fetchone(self):
        return self.cur.fetchone()
        
    def execute_fetchone(self, sql_stmt):
        self.execute_sql(sql_stmt)
        return(self.fetchone())
        
    def fetchmany(self, size=100):
        return(self.cur.fetchmany(size))
        
    def execute_fetchmany(self, sql_stmt, size=100):
        self.execute_sql(sql_stmt)
        return(self.fetchmany(size))
        
    def fetchall(self):
        return self.cur.fetchall()
        
    def execute_fetchall(self, sql_stmt):
        self.execute_sql(sql_stmt)
        return(self.fetchall())
        
    def execute_tiered_query(self, tn, col_names, query_dict_stack,
                             results={},
                             select_sql="""SELECT {cols} \nFROM {tn};\n"""):
        if len(query_dict_stack)<1:
            return(results)
        
        #create single query
        col_sqls = []
        col_query = []
        for cn in col_names:
            if cn not in results:
                results[cn] = {"cn": cn}
                
            for query_name, query_sql in query_dict_stack[0].items():
                col_sqls.append(query_sql.format(**results[cn]))
                col_query.append((cn, query_name))
        
        cols = ",\n".join(col_sqls)
        select_sql = select_sql.format(cols=cols, tn=tn)
        query_res = self.execute_fetchone(select_sql)
        
        #read results
        for val, (cn, query_name) in zip(query_res, col_query):
            results[cn][query_name] = val
        
        return(self.execute_tiered_query(tn, col_names, query_dict_stack[1:],
                                         results))
    



def interactive_session(cur):
    print("Enter your SQL commands to execute in sqlite3.")
    print("Enter a blank line to exit.")

    buffer = ""

    while True:
        line = input()
        if line == "":
            break
        buffer += line

        if sqlite3.complete_statement(buffer):
            try:
                buffer = buffer.strip()
                cur.execute(buffer)

                if buffer.lstrip().upper().startswith("SELECT"):
                    print(cur.fetchall())
            except sqlite3.Error as e:
                print("An error occurred:", e.args[0])
            buffer = ""
            
def execute_sql(cur, sql_stmt, verbose=True, param=None, many=False):
    sql_stmt = sql_stmt.replace("--", "+")
    if verbose:
        print(sql_stmt)
    
    if param is None:
        cur.execute(sql_stmt)
    else:
        if many:
            cur.executemany(sql_stmt, param)
        else:
            cur.execute(sql_stmt, param)

