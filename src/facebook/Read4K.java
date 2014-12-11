package facebook;

import java.util.ArrayList;

public class Read4K {
//	static final int SIZE=4096;
//	ArrayList<Integer> buff;
//	int p;
//	
//	public ArrayList<Integer> read4k(){
//		return new ArrayList<Integer>(4096);
//	}
//	
//	public ArrayList<Integer> readAnySize(int n){
//		ArrayList<Integer> res=new ArrayList<Integer>();
//		while(n>0){
//			if(p+n<SIZE){
//				res.addAll(buff.subList(p, p+n));
//				p+=n;
//				n=0;
//			}
//			else{
//				res.addAll(buff.subList(p, buff.size()));
//				n-=buff.size()-p;
//				p=0;
//				buff=read4k();
//			}
//		}
//		return res;
//	}
	
	String buffer=null;
	int p=0;
	
	public String read4K(){
		return "4096";
	}
	
	public String read(int n){
		if(n<=0)
			return "";
		StringBuilder sb=new StringBuilder();
		while(n>0){
			if(buffer==null||buffer.length()==0){
				buffer=read4K();
				p=0;
				if(buffer.length()==0)//finish reading the file.
					break;
			}
			else{
				int numChars=buffer.length()-p;
				if(numChars>=n){
					sb.append(buffer.substring(p,p+n));
					p+=n;
					n=0;
				}
				else{
					sb.append(buffer.substring(p));
					n-=numChars;
//					buffer=read4K();
//					p=0;
					// or alternative
					p=buffer.length();
				}
			}
		}
		return sb.toString();
	}
	
}
