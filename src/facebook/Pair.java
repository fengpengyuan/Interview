package facebook;

public class Pair {
	int first;
	int second;
	public Pair(int first, int second){
		this.first=first;
		this.second=second;
	}
	
	public String toString(){
		return "("+first+", "+second+")";
	}
}
