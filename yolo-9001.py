
import socket, struct, time
import numpy as np

HEADER = struct.Struct(">IHHIQII")  
MAGIC  = 0x4655534E
PORT   = 9001

def serve(conn):
    buf = bytearray(); lat = []; gaps = []; last_t = None
    while True:
        data = conn.recv(65536)
        if not data:
            return
        t_recv = time.time()                
        buf.extend(data)
        while len(buf) >= 28:
            magic, ver, mtype, plen, ts_us, seq, _ = HEADER.unpack_from(buf, 0)
            if magic != MAGIC:             
                del buf[:4]; continue
            if len(buf) < 28 + plen:
                break
            del buf[:28 + plen]
            lat.append((t_recv - ts_us / 1e6) * 1000.0)
            if last_t is not None:
                gaps.append(t_recv - last_t)
            last_t = t_recv
            if len(lat) % 20 == 0:
                a = np.array(lat[-500:]); g = np.array(gaps[-500:])
                print(f"n={len(lat):5d} |  p50={np.percentile(a,50):6.1f} "
                      f"p90={np.percentile(a,90):6.1f} p99={np.percentile(a,99):6.1f} ms "
                      f"|  {np.median(g)*1000:5.0f} ms")

srv = socket.socket()
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("0.0.0.0", PORT)); srv.listen(1)
print(f"listening 0.0.0.0:{PORT} ...")
while True:
    conn, addr = srv.accept()
    print("connected", addr)
    try:
        serve(conn)
    except OSError as e:
        print("disconnected", e)        
    finally:
        conn.close()