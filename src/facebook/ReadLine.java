package facebook;

public class ReadLine {
	static char[] buf = null;
    int p = 0;
    public char[] read4096(){
    	return new char[4096];
    }
    
    public char[] readLine(){
    	StringBuilder sb=new StringBuilder();
    	if(buf==null)
    		buf=read4096();
    	
    	while(true){
    		if(buf==null||buf.length==0)
    			break;
    		while(p<buf.length){
    			if(buf[p]=='\n'||buf[p]=='\0')
    				return sb.toString().toCharArray();
    			sb.append(buf[p]);
    			p++;
    		}
    		if(buf.length<4096)
    			break;
    		buf=read4096();
    		p=0;
    	}
    	return sb.toString().toCharArray();
    }
}
