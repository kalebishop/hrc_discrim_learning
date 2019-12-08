The Stars2 Corpus of Referring Expressions
University of São Paulo (USP)
Version 2.0 - 13 March 2016.

===============
1. Overview
===============

The Stars2 corpus is a collection of descriptions of 3D geometric shapes (cubes, balls and cones) produced by participants in a controlled experiment for a number of studies on referring expression generation.

The goal of the experiment was to elicit descriptions in situations in which a range of opportunities for target and landmark overspecification is available, including at least one highly salient referential property in each context and significant numbers of relational descriptions involving up to three referents (e.g., "the ball near the cube, next to the large red sphere").

The descriptions were originally written in Brazilian Portuguese (presently accompanied by their English translation for illustration purposes) and subsequently annotated with their semantic properties as attribute-value pairs, all of which represented in XML format.

Details of the Stars2 contents are described in Paraboni, Galindo and Iacovelli (2016) "Stars2: a Corpus of Object Descriptions in a Visual Domain", to appear in Language Resources and Evaluation (DOI 10.1007/s10579-016-9350-y).

=======
2. Data
=======

The Stars2 corpus consists of 884 descriptions produced by 56 participants in 16 visual contexts each. The following data sets are provided:

The [description] folder contains the actual description sets produced by each of the 56 speaker and their annotation as attribute sets. The attribute sets are represented both as a series of XML nodes and also as a single ANNOTATION tag for ease of processing. Details of the annotation scheme are described in the above publication. 

The file [Stars2-contexts.xml] is the XML representation of the 64 possible scenes used in the experiment. Each scene is identified by an ID tag discussed below.

The [images] folder contains the 64 possible scenes seen by the speakers in the experiment (out of which 16 were selected as discussed in the paper). The target in each scene is identified by an arrow (which was visible only to the speaker). The present images do not show object labels, which were only visible to the hearers. For the actual labels assigned to each object, see the context definition in the [Stars2-contexts.xml] file. Image filenames correspond to the context ID referred to in each corpus descriptions as discussed below.


===================================
3. Context IDs and image filenames
====================================

Each scene is assigned a context ID that is referred to both in the context definition file (Stars2-contexts.xml), in the description files (in the descriptions folder) and also represented in the image filenames. A context ID is to interpreted as follows:

* The first two digits of the context ID identify the experiment condition representing a context 01..08 as described in the paper. In contexts 01..04 the target object is not unique, and the use of a relational description is most likely required. In contexts 05..08, by contrast, the target type is unique, and an atomic description is possible.

* The third character is a letter indicating whether the scene was presented in flat (f) or overlapping (o) mode, as illustrated in the paper.

* The t1/t2 identifier denotes the order in which object types appear in each scene. The images in folder type1 represent the same contexts (01..08) as those in type2 except for the object order. 

* A letter "n" following the object order t1/t2 denotes normal orientation, and a letter "r" denotes reverse (i.e., mirrored) image orientation.


======================
4. How to cite Stars2
======================

If you wish to use Stars2 for research purposes, please cite the following publications:

Paraboni, Galindo and Iacovelli (2016) 
Stars2: a Corpus of Object Descriptions in a Visual Domain
Language Resources and Evaluation (DOI 10.1007/s10579-016-9350-y).


======================
5. Further information
======================

For further information, please feel free to contact the authors:

Ivandré Paraboni
ivandre  @  usp . br
School of Arts, Sciences and Humanities (EACH)
University of São Paulo (USP)
São Paulo, Brazil
