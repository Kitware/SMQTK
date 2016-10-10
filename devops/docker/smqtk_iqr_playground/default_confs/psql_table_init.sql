CREATE TABLE IF NOT EXISTS descriptors_alexnet_fc7 (
  type_str  TEXT  NOT NULL,
  uid       TEXT  NOT NULL,
  vector    BYTEA NOT NULL,

  PRIMARY KEY (uid, type_str)
);
