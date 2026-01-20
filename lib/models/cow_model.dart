class Cow {
  final String name;
  final String id;
  final String status;
  final String temp;
  final String heart;
  final String activity;
  final String image;

  Cow({
    required this.name,
    required this.id,
    required this.status,
    required this.temp,
    required this.heart,
    required this.activity,
    required this.image,
  });

  factory Cow.fromJson(Map<String, dynamic> json) {
    return Cow(
      name: json['name'] ?? '',
      id: json['id']?.toString() ?? '',
      status: json['status'] ?? '',
      temp: json['temp']?.toString() ?? '',
      heart: json['heart']?.toString() ?? '',
      activity: json['activity'] ?? '',
      image: json['image'] ?? '',
    );
  }

  // tambahan: buat dari DocumentSnapshot Firestore
  factory Cow.fromDocument(dynamic doc) {
    final data = (doc.data() is Map) ? Map<String, dynamic>.from(doc.data() as Map) : <String, dynamic>{};
    data['id'] = data['id'] ?? (doc.id ?? '');
    return Cow.fromJson(data);
  }

  // tambahan: konversi ke Map untuk update/penyimpanan
  Map<String, dynamic> toJson() {
    return {
      'name': name,
      'id': id,
      'status': status,
      'temp': temp,
      'heart': heart,
      'activity': activity,
      'image': image,
    };
  }
}