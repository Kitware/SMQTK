DROP TABLE IF EXISTS descriptors;
CREATE TABLE IF NOT EXISTS descriptors (
  type_str  TEXT  NOT NULL,
  uid       TEXT  NOT NULL,
  vector    BYTEA NOT NULL,

  PRIMARY KEY (uid, type_str)
);
