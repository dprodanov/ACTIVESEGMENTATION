package activeSegmentation;

/*
 * Enumeration for different filter types
 * 	SEGM - segmentation: one to one NxN -> NxN
	CLASSIF - segmentation: many to one NxN -> 1
 */
public enum FilterType {

	SEGM(1),  
	CLASSIF(2);
  
	private final int filterType;

	FilterType(int ft) {
		this.filterType = ft;
	}

	public int getFilterType() {
		return filterType;
	}
}