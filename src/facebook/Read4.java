package facebook;

public class Read4 {
	int read4(char[] buff){
		return 4;
	}
	
	/**
	    * @param buf Destination buffer
	    * @param n   Maximum number of characters to read
	    * @return    The number of characters read
	    */
	   public int read(char[] buf, int n) {
		   char[] buffer=new char[4];
		   int readSize=0;
		   boolean eof=false;
		   while(!eof&&readSize<n){
			   int size=read4(buffer);
			   if(size<4)
				   eof=true;
			   int bytes=Math.min(n-readSize, size);
			   System.arraycopy(buffer, 0, buf, readSize, bytes);
			   readSize+=bytes;
		   }
		   return readSize;
	   }
}
