CREATE TABLE IF NOT EXISTS descriptors_resnet50_pool5 (
  type_str  TEXT  NOT NULL,
  uid       TEXT  NOT NULL,
  vector    BYTEA NOT NULL,

  PRIMARY KEY (uid, type_str)
);
CREATE TABLE IF NOT EXISTS descriptor_index_resnet50_pool5 (
  uid       TEXT  NOT NULL,
  element   BYTEA NOT NULL,

  PRIMARY KEY (uid)
);
